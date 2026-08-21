
# Deployment

The API ships as a single container image. The same image runs unchanged on
Google Cloud Run, Azure Container Apps, and AWS App Runner — all three inject
`PORT` and expect the container to listen on it, which is why the Dockerfile
reads it from the environment rather than hardcoding 8000.

Build once, deploy anywhere. That portability is the point, and it is what
makes the choice of provider a cost and latency decision rather than an
engineering one.

---

## What is in the image

|            |                                                                                 |
| :--------- | :------------------------------------------------------------------------------ |
| Models     | XGBoost at 6h, 12h, 24h (~28 MB)                                                |
| Excluded   | Random Forest (~633 MB for the same three horizons, no accuracy advantage), training dependencies, Streamlit |
| Base       | `python:3.11-slim`                                                            |
| Final size | ~400 MB                                                                         |

Random Forest is left out because XGBoost matches or beats it at every horizon
and the 72h difference (501.2 vs 505.4 km) is inside the confidence interval,
while the artifacts are over twenty times larger. Serving both would mean a
~660 MB heavier image for no measurable gain.

`available_models()` scans the models directory at runtime, so `/health`
reports exactly what shipped and `/predict` returns 503 with a clear message
for any horizon that did not. Adding a horizon means adding two lines to the
Dockerfile, not changing code.

---

## Build and test locally

```bash
docker build -f Dockerfile.deploy -t cyclone-api .
docker run --rm -p 8000:8000 cyclone-api
```

Check it before pushing anywhere:

```bash
curl localhost:8000/health
# {"status":"healthy","models_available":{"xgboost":[6,12,24]},...}
```

`"degraded"` means the models did not make it into the image — check that
`models/xgboost_*.pkl` exist locally and are not excluded by `.dockerignore`.

Interactive docs at http://localhost:8000/docs. That Swagger page is the demo:
it lets anyone submit five observations and get a forecast without writing a
line of code.

---

## Google Cloud Run

The lowest-friction of the three. Scales to zero, so an idle demo costs
nothing, and the free tier covers two million requests a month.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

gcloud run deploy cyclone-api \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8000
```

`--source .` builds with Cloud Build, so no local Docker is needed. It picks
up `Dockerfile` by default; rename `Dockerfile.deploy` or pass
`--docker-file` if your gcloud version supports it.

`--min-instances 0` is what makes it free, at the cost of a cold start of
roughly ten to fifteen seconds on the first request after idling. For a demo
that is the right trade; for anything real, `--min-instances 1`.

`asia-south1` is Mumbai. Pick the region nearest your users.

---

## Azure Container Apps

The closest Azure equivalent — also scales to zero, also KNative-derived.

```bash
az login
az group create --name cyclone-rg --location centralindia

az acr create --resource-group cyclone-rg --name cycloneacr --sku Basic
az acr login --name cycloneacr

docker build -f Dockerfile.deploy -t cycloneacr.azurecr.io/cyclone-api:v1 .
docker push cycloneacr.azurecr.io/cyclone-api:v1

az containerapp up \
  --name cyclone-api \
  --resource-group cyclone-rg \
  --image cycloneacr.azurecr.io/cyclone-api:v1 \
  --target-port 8000 \
  --ingress external \
  --registry-server cycloneacr.azurecr.io \
  --min-replicas 0 \
  --max-replicas 3
```

Azure Container Registry Basic is not free (~$5/month). Delete the resource
group when you are finished demonstrating it:

```bash
az group delete --name cyclone-rg --yes
```

---

## AWS App Runner

Fewest knobs of the three, but no scale-to-zero — it bills continuously while
a service exists.

```bash
aws ecr create-repository --repository-name cyclone-api --region ap-south-1

aws ecr get-login-password --region ap-south-1 \
  | docker login --username AWS --password-stdin \
    <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com

docker build -f Dockerfile.deploy -t cyclone-api .
docker tag cyclone-api:latest <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/cyclone-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/cyclone-api:latest

aws apprunner create-service \
  --service-name cyclone-api \
  --source-configuration file://apprunner.json
```

`apprunner.json`:

```json
{
  "ImageRepository": {
    "ImageIdentifier": "<ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/cyclone-api:latest",
    "ImageConfiguration": { "Port": "8000" },
    "ImageRepositoryType": "ECR"
  },
  "AutoDeploymentsEnabled": false
}
```

Roughly $5–7/month at the smallest size, running or not. Delete it when the
demo is over.

---

## Dashboard

The Streamlit dashboard is a separate deployment that calls the API over HTTP.
Streamlit Community Cloud is free and deploys from GitHub:

1. Push the repo to GitHub.
2. At share.streamlit.io, point a new app at `streamlit_app.py`.
3. Under **Secrets**, set:

```toml
API_URL = "https://your-api-url"
```

Keeping the two separate means only the API container loads models, and the
dashboard is a thin client. Bundling them would put two processes in one
container and load every model twice.

---

## Which to use

For a portfolio project, **Cloud Run**. It scales to zero, so an idle demo is
free; the deploy is one command; and the URL stays live indefinitely without
accruing cost.

Azure and AWS are worth doing once each if you want to say you have. The image
is identical, so the only difference is the registry and the deploy command —
which is itself the useful observation: containerising the application is the
engineering, and the provider is a procurement decision.

---

## Honest notes for discussion

**Models are baked into the image.** At portfolio scale, committing 9 MB of
XGBoost artifacts is pragmatic. At real scale they belong in a model registry
(MLflow, Vertex AI Model Registry, SageMaker) pulled at startup, so a model
update does not require an image rebuild and a rollback is a version change
rather than a redeploy.

**No authentication.** `--allow-unauthenticated` makes the demo shareable and
would be wrong for anything real. The API is read-only and stateless, so the
exposure is limited to compute cost.

**One worker.** `--workers 1` with a single CPU. Each worker loads its own copy
of every model, so scaling means more container instances rather than more
workers per instance.

**No request logging or metrics.** A production deployment would emit
structured logs to Cloud Logging or Application Insights and track prediction
latency and input distribution — the latter to detect the drift that a track
model would eventually suffer as the observation network changes.
