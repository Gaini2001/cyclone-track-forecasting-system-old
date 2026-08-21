# Results

_Generated 2026-08-21 — target mode: `displacement`_

Track error is great-circle distance between forecast and observed position. Skill is the percentage reduction in mean error relative to persistence, with a paired bootstrap confidence interval; `p` is the bootstrap fraction in which the forecast failed to beat persistence.

## Headline comparison

| Model                |   Horizon (h) |   Mean Error (km) | 95% CI    |   Median (km) |   P90 (km) |   Within 100km (%) |   R2 (displacement) |   Cross-track MAE (km) |   Along-track MAE (km) |   Along-track bias (km) | Skill vs Persistence (%)   | Skill 95% CI   | p   |    n |
|:---------------------|--------------:|------------------:|:----------|--------------:|-----------:|-------------------:|--------------------:|-----------------------:|-----------------------:|------------------------:|:---------------------------|:---------------|:----|-----:|
| XGBoost              |             6 |              11.3 | 11–12     |           9.5 |       20.5 |              100   |               0.983 |                    6.7 |                    7.7 |                    -1.2 | +89.3                      | +89.1–+89.5    | 0.0 | 5300 |
| Random Forest        |             6 |              13.6 | 13–14     |          11.1 |       26.2 |              100   |               0.975 |                    8.1 |                    9.2 |                    -1.6 | +87.2                      | +86.9–+87.4    | 0.0 | 5300 |
| Linear Regression    |             6 |              24.2 | 24–25     |          19.2 |       47.2 |               99.1 |               0.919 |                   13.1 |                   17.4 |                    -2.9 | +77.2                      | +76.6–+77.6    | 0.0 | 5300 |
| Linear Extrapolation |             6 |              29.2 | 29–30     |          24.2 |       56.6 |               98.7 |               0.889 |                   17.5 |                   19.5 |                    -6.1 | +72.5                      | +71.9–+73.0    | 0.0 | 5300 |
| Climatology          |             6 |              81.7 | 80–83     |          70.8 |      150.1 |               72.1 |               0.253 |                   34.5 |                   69.9 |                   -66.5 | +23.0                      | +22.2–+23.9    | 0.0 | 5300 |
| Persistence          |             6 |             106.1 | 105–108   |          97.9 |      174.7 |               51.9 |              -0.149 |                    0.7 |                  106.3 |                  -106.3 | ref                        | —              | —   | 5300 |
| XGBoost              |            12 |              40.1 | 39–41     |          33.7 |       75.3 |               95.7 |               0.945 |                   24.3 |                   26.7 |                    -7.8 | +80.8                      | +80.4–+81.2    | 0.0 | 5085 |
| Random Forest        |            12 |              42.8 | 42–44     |          35.8 |       80.7 |               94.7 |               0.937 |                   26.2 |                   28.3 |                    -8.2 | +79.5                      | +79.0–+79.9    | 0.0 | 5085 |
| Linear Regression    |            12 |              56.4 | 55–58     |          46.5 |      108.7 |               87.5 |               0.89  |                   32.9 |                   38.3 |                   -10.2 | +73.0                      | +72.4–+73.5    | 0.0 | 5085 |
| Linear Extrapolation |            12 |              65.8 | 64–67     |          54.8 |      127.4 |               81.9 |               0.854 |                   40.3 |                   43.3 |                   -14.3 | +68.4                      | +67.8–+69.1    | 0.0 | 5085 |
| Climatology          |            12 |             157.8 | 155–161   |         138   |      289.2 |               32   |               0.267 |                   68.7 |                  133.8 |                  -127   | +24.3                      | +23.4–+25.1    | 0.0 | 5085 |
| Persistence          |            12 |             208.4 | 205–212   |         193   |      340.1 |               14.5 |              -0.161 |                    2.5 |                  208.5 |                  -208.5 | ref                        | —              | —   | 5085 |
| XGBoost              |            24 |             116.4 | 114–119   |          99.1 |      215   |               50.6 |               0.874 |                   70.1 |                   77.6 |                   -36.7 | +71.2                      | +70.5–+71.8    | 0.0 | 4657 |
| Random Forest        |            24 |             120   | 118–122   |         102.1 |      221.4 |               48.5 |               0.867 |                   72.8 |                   79.6 |                   -36.4 | +70.3                      | +69.6–+70.9    | 0.0 | 4657 |
| Linear Regression    |            24 |             139.3 | 137–142   |         117.5 |      267.3 |               40.5 |               0.82  |                   84.2 |                   92.5 |                   -38.8 | +65.5                      | +64.7–+66.2    | 0.0 | 4657 |
| Linear Extrapolation |            24 |             160.1 | 157–163   |         133.3 |      303.2 |               33.3 |               0.762 |                   98.9 |                  105.1 |                   -38.6 | +60.3                      | +59.5–+61.2    | 0.0 | 4657 |
| Climatology          |            24 |             297.4 | 292–303   |         260   |      546.5 |               11.3 |               0.29  |                  135.7 |                  248.5 |                  -234.5 | +26.3                      | +25.4–+27.2    | 0.0 | 4657 |
| Persistence          |            24 |             403.4 | 398–409   |         379   |      660.2 |                3.2 |              -0.186 |                    9   |                  403.2 |                  -403.2 | ref                        | —              | —   | 4657 |
| XGBoost              |            48 |             307   | 301–313   |         267   |      563.9 |                9.8 |               0.751 |                  176.6 |                  214.5 |                  -137.2 | +59.7                      | +58.9–+60.6    | 0.0 | 3794 |
| Random Forest        |            48 |             315.1 | 309–321   |         270   |      584   |                9.2 |               0.739 |                  185.2 |                  216.6 |                  -134.7 | +58.6                      | +57.8–+59.6    | 0.0 | 3794 |
| Linear Regression    |            48 |             335.3 | 328–342   |         290.6 |      603.4 |                8.1 |               0.706 |                  198.3 |                  228.8 |                  -138.4 | +56.0                      | +55.0–+57.0    | 0.0 | 3794 |
| Linear Extrapolation |            48 |             403.1 | 395–411   |         346.3 |      753.6 |                7   |               0.564 |                  248   |                  264.3 |                   -96.7 | +47.1                      | +45.9–+48.4    | 0.0 | 3794 |
| Climatology          |            48 |             534.9 | 525–545   |         479.9 |      970   |                2.9 |               0.333 |                  262.4 |                  434   |                  -403.7 | +29.8                      | +28.9–+30.9    | 0.0 | 3794 |
| Persistence          |            48 |             762   | 751–773   |         719.5 |     1248.3 |                0.7 |              -0.235 |                   29.7 |                  761.1 |                  -761.1 | ref                        | —              | —   | 3794 |
| XGBoost              |            72 |             501.2 | 489–512   |         443.3 |      888   |                3.8 |               0.677 |                  278.4 |                  363.7 |                  -282.4 | +54.1                      | +53.0–+55.2    | 0.0 | 2933 |
| Random Forest        |            72 |             505.4 | 493–517   |         446.6 |      914.9 |                4.4 |               0.668 |                  303.5 |                  346   |                  -246.3 | +53.7                      | +52.5–+54.9    | 0.0 | 2933 |
| Linear Regression    |            72 |             527.6 | 516–539   |         463.3 |      935.5 |                3.4 |               0.645 |                  314.3 |                  365.9 |                  -260.6 | +51.7                      | +50.5–+52.9    | 0.0 | 2933 |
| Linear Extrapolation |            72 |             686.6 | 670–702   |         596.7 |     1290.3 |                2   |               0.374 |                  429   |                  447   |                  -155.7 | +37.1                      | +35.4–+39.0    | 0.0 | 2933 |
| Climatology          |            72 |             730.6 | 715–747   |         660.1 |     1315.3 |                1.9 |               0.371 |                  375.8 |                  579.1 |                  -530.1 | +33.1                      | +31.9–+34.2    | 0.0 | 2933 |
| Persistence          |            72 |            1091.8 | 1073–1110 |        1032   |     1794.6 |                0.2 |              -0.283 |                   59.1 |                 1089.6 |                 -1089.6 | ref                        | —              | —   | 2933 |

## Mean track error by lead time (km)

| Model                |     6 |    12 |    24 |    48 |     72 |
|:---------------------|------:|------:|------:|------:|-------:|
| Climatology          |  81.7 | 157.8 | 297.4 | 534.9 |  730.6 |
| Linear Extrapolation |  29.2 |  65.8 | 160.1 | 403.1 |  686.6 |
| Linear Regression    |  24.2 |  56.4 | 139.3 | 335.3 |  527.6 |
| Persistence          | 106.1 | 208.4 | 403.4 | 762   | 1091.8 |
| Random Forest        |  13.6 |  42.8 | 120   | 315.1 |  505.4 |
| XGBoost              |  11.3 |  40.1 | 116.4 | 307   |  501.2 |

## By basin

| model                |   horizon | basin   |   n_samples |   mean_km |   median_km |   cross_track_mae_km |   along_track_bias_km |
|:---------------------|----------:|:--------|------------:|----------:|------------:|---------------------:|----------------------:|
| Persistence          |         6 | SI      |        1216 |      86.5 |        77.6 |                  0.4 |                 -87   |
| Persistence          |         6 | SP      |         341 |     111.3 |        89.6 |                  0.9 |                -111.3 |
| Persistence          |         6 | WP      |        1148 |     104.5 |        96   |                  0.6 |                -104.6 |
| Persistence          |         6 | NI      |         279 |      73.8 |        70.1 |                  0.2 |                 -74.1 |
| Persistence          |         6 | EP      |         900 |     114.2 |       109.9 |                  0.6 |                -114.2 |
| Persistence          |         6 | NA      |        1416 |     124.2 |       116.6 |                  1.1 |                -124.2 |
| Linear Extrapolation |         6 | SI      |        1216 |      32.9 |        24.7 |                 19.5 |                  -6.7 |
| Linear Extrapolation |         6 | SP      |         341 |      40   |        34.1 |                 22.1 |                 -10.8 |
| Linear Extrapolation |         6 | WP      |        1148 |      31.8 |        28.7 |                 19   |                  -6.7 |
| Linear Extrapolation |         6 | NI      |         279 |      29.7 |        24.6 |                 16.1 |                  -6.4 |
| Linear Extrapolation |         6 | EP      |         900 |      18.7 |        15.4 |                 11.5 |                  -1.7 |
| Linear Extrapolation |         6 | NA      |        1416 |      28   |        22.7 |                 17.7 |                  -6.7 |
| Climatology          |         6 | SI      |        1216 |      73.2 |        63.7 |                 29   |                 -61.3 |
| Climatology          |         6 | SP      |         341 |     101   |        83.2 |                 37.2 |                 -86.3 |
| Climatology          |         6 | WP      |        1148 |      81.3 |        71.8 |                 39   |                 -61.6 |
| Climatology          |         6 | NI      |         279 |      60.8 |        53.3 |                 25.3 |                 -50.5 |
| Climatology          |         6 | EP      |         900 |      62.7 |        57.8 |                 27.1 |                 -44.9 |
| Climatology          |         6 | NA      |        1416 |     100.8 |        88.6 |                 41.4 |                 -87   |
| Linear Regression    |         6 | SI      |        1216 |      28.9 |        23.1 |                 15.1 |                  -1.9 |
| Linear Regression    |         6 | SP      |         341 |      37.5 |        31.1 |                 18.1 |                  -3.3 |
| Linear Regression    |         6 | WP      |        1148 |      24.7 |        21.1 |                 13.5 |                  -3.5 |
| Linear Regression    |         6 | NI      |         279 |      24.5 |        20.3 |                 12.2 |                  -2.5 |
| Linear Regression    |         6 | EP      |         900 |      15.6 |        13.6 |                  8.8 |                  -2.6 |
| Linear Regression    |         6 | NA      |        1416 |      22   |        17.1 |                 12.7 |                  -3.3 |
| Random Forest        |         6 | SI      |        1216 |      14.5 |        12   |                  8.8 |                  -1.8 |
| Random Forest        |         6 | SP      |         341 |      18.6 |        15.5 |                 11   |                  -2   |
| Random Forest        |         6 | WP      |        1148 |      13.7 |        11.6 |                  8.3 |                  -1.5 |
| Random Forest        |         6 | NI      |         279 |      16.1 |        14.3 |                  8.3 |                  -1.8 |
| Random Forest        |         6 | EP      |         900 |       9.2 |         8.5 |                  5.7 |                  -0.4 |
| Random Forest        |         6 | NA      |        1416 |      13.8 |        11   |                  8.2 |                  -2.3 |
| XGBoost              |         6 | SI      |        1216 |      11.3 |        10   |                  6.7 |                  -1.6 |
| XGBoost              |         6 | SP      |         341 |      14.6 |        12.4 |                  8.4 |                  -1.5 |
| XGBoost              |         6 | WP      |        1148 |      11.6 |        10.2 |                  7   |                  -1   |
| XGBoost              |         6 | NI      |         279 |      15.5 |        13.4 |                  8   |                  -2.3 |
| XGBoost              |         6 | EP      |         900 |       8.1 |         7.5 |                  5   |                  -0.8 |
| XGBoost              |         6 | NA      |        1416 |      11.6 |         9.2 |                  6.8 |                  -1.1 |
| Persistence          |        12 | SI      |        1164 |     169   |       154.7 |                  1.5 |                -169.1 |
| Persistence          |        12 | SP      |         327 |     217.6 |       173.4 |                  3.4 |                -218.2 |
| Persistence          |        12 | WP      |        1101 |     204.1 |       190.9 |                  2.2 |                -204.1 |
| Persistence          |        12 | NI      |         268 |     144.2 |       133.4 |                  0.7 |                -144.2 |
| Persistence          |        12 | EP      |         863 |     227.7 |       221   |                  2.4 |                -227.6 |
| Persistence          |        12 | NA      |        1362 |     243.8 |       232.2 |                  4   |                -243.8 |
| Linear Extrapolation |        12 | SI      |        1164 |      69.8 |        56.6 |                 41.5 |                 -14.2 |
| Linear Extrapolation |        12 | SP      |         327 |      84.9 |        75.7 |                 48   |                 -26.5 |
| Linear Extrapolation |        12 | WP      |        1101 |      69.4 |        61.8 |                 42.5 |                 -15.5 |
| Linear Extrapolation |        12 | NI      |         268 |      60.4 |        54.8 |                 33.6 |                 -11.9 |
| Linear Extrapolation |        12 | EP      |         863 |      44.5 |        39.6 |                 28.1 |                  -4.5 |
| Linear Extrapolation |        12 | NA      |        1362 |      69.4 |        55.6 |                 44.9 |                 -17.3 |
| Climatology          |        12 | SI      |        1164 |     141.2 |       123.8 |                 58.1 |                -116.5 |
| Climatology          |        12 | SP      |         327 |     196.4 |       161.9 |                 74.8 |                -166.8 |
| Climatology          |        12 | WP      |        1101 |     155.1 |       140   |                 77.4 |                -115.2 |
| Climatology          |        12 | NI      |         268 |     116   |       104.2 |                 49.2 |                 -95.7 |
| Climatology          |        12 | EP      |         863 |     123.6 |       113.1 |                 54   |                 -87.8 |
| Climatology          |        12 | NA      |        1362 |     194.9 |       169.4 |                 82.5 |                -167.1 |
| Linear Regression    |        12 | SI      |        1164 |      63.8 |        52.1 |                 36.4 |                  -9.3 |
| Linear Regression    |        12 | SP      |         327 |      83.3 |        72.8 |                 44.8 |                 -13.7 |
| Linear Regression    |        12 | WP      |        1101 |      56.1 |        48.3 |                 32.7 |                 -10.9 |
| Linear Regression    |        12 | NI      |         268 |      52.8 |        46.9 |                 29.8 |                  -7.8 |
| Linear Regression    |        12 | EP      |         863 |      38.8 |        36.5 |                 23.2 |                  -8.6 |
| Linear Regression    |        12 | NA      |        1362 |      55.6 |        44.2 |                 34.1 |                 -11.2 |
| Random Forest        |        12 | SI      |        1164 |      45.6 |        37.6 |                 27.4 |                  -8.6 |
| Random Forest        |        12 | SP      |         327 |      55.5 |        46.9 |                 33.3 |                 -11.2 |
| Random Forest        |        12 | WP      |        1101 |      42.4 |        36.6 |                 26.6 |                  -7.9 |
| Random Forest        |        12 | NI      |         268 |      44.5 |        40.8 |                 24.5 |                  -6.4 |
| Random Forest        |        12 | EP      |         863 |      30.6 |        27.5 |                 19.5 |                  -5.1 |
| Random Forest        |        12 | NA      |        1362 |      45.1 |        36.1 |                 27.9 |                  -9.5 |
| XGBoost              |        12 | SI      |        1164 |      42.2 |        36.3 |                 25   |                 -10   |
| XGBoost              |        12 | SP      |         327 |      53.7 |        45.4 |                 31.3 |                 -11.1 |
| XGBoost              |        12 | WP      |        1101 |      39.5 |        35.2 |                 24.6 |                  -5.9 |
| XGBoost              |        12 | NI      |         268 |      43.2 |        38.3 |                 24   |                  -9.1 |
| XGBoost              |        12 | EP      |         863 |      28.3 |        25.5 |                 17.9 |                  -5.2 |
| XGBoost              |        12 | NA      |        1362 |      42.3 |        33.9 |                 25.8 |                  -7.9 |
| Persistence          |        24 | SI      |        1062 |     324.5 |       300.9 |                  5.3 |                -324.4 |
| Persistence          |        24 | SP      |         299 |     420.5 |       334.5 |                 12.6 |                -420.2 |
| Persistence          |        24 | WP      |        1007 |     391.6 |       373.3 |                  7.5 |                -391.5 |
| Persistence          |        24 | NI      |         246 |     277.8 |       267.7 |                  2.6 |                -277.8 |
| Persistence          |        24 | EP      |         789 |     452.2 |       446.2 |                  9.6 |                -452.1 |
| Persistence          |        24 | NA      |        1254 |     469.5 |       450.7 |                 13.3 |                -469.1 |
| Linear Extrapolation |        24 | SI      |        1062 |     161.4 |       131.5 |                 95   |                 -35.5 |
| Linear Extrapolation |        24 | SP      |         299 |     207.2 |       179.2 |                108.5 |                 -81.6 |
| Linear Extrapolation |        24 | WP      |        1007 |     165.6 |       145.1 |                105.4 |                 -39.4 |
| Linear Extrapolation |        24 | NI      |         246 |     132.8 |       118.9 |                 76.6 |                 -35.7 |
| Linear Extrapolation |        24 | EP      |         789 |     112   |       101.3 |                 70.2 |                 -13.8 |
| Linear Extrapolation |        24 | NA      |        1254 |     179.2 |       146.1 |                117   |                 -46.5 |
| Climatology          |        24 | SI      |        1062 |     267   |       233.1 |                116.2 |                -215.6 |
| Climatology          |        24 | SP      |         299 |     376   |       307.9 |                148.8 |                -313.1 |
| Climatology          |        24 | WP      |        1007 |     286.5 |       264.5 |                151.3 |                -204.8 |
| Climatology          |        24 | NI      |         246 |     220.1 |       204.9 |                 98.2 |                -181.6 |
| Climatology          |        24 | EP      |         789 |     240.6 |       222.2 |                107.5 |                -168   |
| Climatology          |        24 | NA      |        1254 |     364   |       318.8 |                161.7 |                -307.9 |
| Linear Regression    |        24 | SI      |        1062 |     150.6 |       128.8 |                 89.7 |                 -34.5 |
| Linear Regression    |        24 | SP      |         299 |     202.8 |       186.7 |                107.6 |                 -63.8 |
| Linear Regression    |        24 | WP      |        1007 |     136.1 |       120.3 |                 85.6 |                 -37.8 |
| Linear Regression    |        24 | NI      |         246 |     116.6 |        99.3 |                 67.9 |                 -36.7 |
| Linear Regression    |        24 | EP      |         789 |     102.2 |        94   |                 61.7 |                 -30.3 |
| Linear Regression    |        24 | NA      |        1254 |     144.9 |       118.2 |                 90.2 |                 -43   |
| Random Forest        |        24 | SI      |        1062 |     122.4 |       106.4 |                 71.6 |                 -40.2 |
| Random Forest        |        24 | SP      |         299 |     157.5 |       145.6 |                 82.7 |                 -62.1 |
| Random Forest        |        24 | WP      |        1007 |     116.7 |       102.3 |                 76   |                 -29.4 |
| Random Forest        |        24 | NI      |         246 |     108.5 |        96   |                 63.2 |                 -25.3 |
| Random Forest        |        24 | EP      |         789 |      91.3 |        81.2 |                 56   |                 -26.5 |
| Random Forest        |        24 | NA      |        1254 |     131.9 |       109.4 |                 81.4 |                 -41   |
| XGBoost              |        24 | SI      |        1062 |     118.3 |       103.1 |                 68.8 |                 -43.1 |
| XGBoost              |        24 | SP      |         299 |     158.2 |       145.1 |                 80.5 |                 -72.9 |
| XGBoost              |        24 | WP      |        1007 |     113.2 |       101.8 |                 72.5 |                 -25   |
| XGBoost              |        24 | NI      |         246 |     109   |        95.5 |                 62.6 |                 -32.3 |
| XGBoost              |        24 | EP      |         789 |      86.8 |        77.9 |                 53.7 |                 -25.1 |
| XGBoost              |        24 | NA      |        1254 |     127.3 |       104.3 |                 78.5 |                 -40.1 |
| Persistence          |        48 | SI      |         855 |     608.5 |       577.6 |                 17.8 |                -608.2 |
| Persistence          |        48 | SP      |         243 |     785.1 |       642.7 |                 41.4 |                -783.4 |
| Persistence          |        48 | WP      |         819 |     724.9 |       699   |                 23.5 |                -724.3 |
| Persistence          |        48 | EP      |         641 |     895.7 |       885.5 |                 37.9 |                -894.7 |
| Persistence          |        48 | NA      |        1038 |     876.9 |       862.4 |                 40.8 |                -875.4 |
| Linear Extrapolation |        48 | SI      |         855 |     393.9 |       350.3 |                224.6 |                 -75.6 |
| Linear Extrapolation |        48 | SP      |         243 |     544.7 |       500   |                249.4 |                -280.8 |
| Linear Extrapolation |        48 | WP      |         819 |     414.8 |       373.8 |                269.4 |                 -88.6 |
| Linear Extrapolation |        48 | EP      |         641 |     279.3 |       257.7 |                180.3 |                 -36.3 |
| Linear Extrapolation |        48 | NA      |        1038 |     461.7 |       390   |                303.8 |                -116.1 |
| Climatology          |        48 | SI      |         855 |     488.1 |       426.5 |                231.8 |                -376.7 |
| Climatology          |        48 | SP      |         243 |     689   |       558.3 |                304.5 |                -557.7 |
| Climatology          |        48 | WP      |         819 |     496.6 |       467.7 |                287   |                -324.6 |
| Climatology          |        48 | EP      |         641 |     458.2 |       422.3 |                207.5 |                -307.9 |
| Climatology          |        48 | NA      |        1038 |     641.2 |       567.2 |                306.4 |                -527.6 |
| Linear Regression    |        48 | SI      |         855 |     343.6 |       311   |                199.5 |                -122.8 |
| Linear Regression    |        48 | SP      |         243 |     495.3 |       448.1 |                220.3 |                -259.9 |
| Linear Regression    |        48 | WP      |         819 |     333.4 |       301.2 |                213.3 |                -125.8 |
| Linear Regression    |        48 | EP      |         641 |     254.5 |       230.2 |                151.4 |                -111.1 |
| Linear Regression    |        48 | NA      |        1038 |     356.8 |       298.3 |                219.5 |                -149.3 |
| Random Forest        |        48 | SI      |         855 |     308.1 |       274.9 |                174   |                -140.6 |
| Random Forest        |        48 | SP      |         243 |     460.5 |       426.4 |                192.8 |                -285   |
| Random Forest        |        48 | WP      |         819 |     309.8 |       273.9 |                203.5 |                 -97.2 |
| Random Forest        |        48 | EP      |         641 |     240.2 |       217.6 |                143.4 |                 -96.1 |
| Random Forest        |        48 | NA      |        1038 |     344.3 |       292.6 |                202.7 |                -160.4 |
| XGBoost              |        48 | SI      |         855 |     302.2 |       270.1 |                165.6 |                -146.2 |
| XGBoost              |        48 | SP      |         243 |     464.7 |       402.4 |                183.4 |                -312.3 |
| XGBoost              |        48 | WP      |         819 |     299.2 |       270.8 |                192.6 |                 -96.5 |
| XGBoost              |        48 | EP      |         641 |     233.7 |       207.8 |                134.2 |                -107.2 |
| XGBoost              |        48 | NA      |        1038 |     333.2 |       286.8 |                199.7 |                -146.3 |
| Persistence          |        72 | SI      |         650 |     877.9 |       852.3 |                 36.7 |                -876.9 |
| Persistence          |        72 | WP      |         631 |    1008.5 |       959.6 |                 42.6 |               -1007.3 |
| Persistence          |        72 | EP      |         493 |    1341.8 |      1299.2 |                 86.7 |               -1338.6 |
| Persistence          |        72 | NA      |         822 |    1243.6 |      1246.8 |                 77.4 |               -1240.2 |
| Linear Extrapolation |        72 | SI      |         650 |     674.4 |       594.1 |                392.7 |                -130.1 |
| Linear Extrapolation |        72 | WP      |         631 |     696.7 |       633   |                459.8 |                -129.2 |
| Linear Extrapolation |        72 | EP      |         493 |     474   |       404.5 |                292.1 |                 -59.8 |
| Linear Extrapolation |        72 | NA      |         822 |     782.6 |       686.3 |                535.5 |                -165.6 |
| Climatology          |        72 | SI      |         650 |     685.9 |       585   |                348.2 |                -515   |
| Climatology          |        72 | WP      |         631 |     652.9 |       624.6 |                405.8 |                -387.7 |
| Climatology          |        72 | EP      |         493 |     659.3 |       614.6 |                289.5 |                -426.1 |
| Climatology          |        72 | NA      |         822 |     855.4 |       754.4 |                428.9 |                -680.7 |
| Linear Regression    |        72 | SI      |         650 |     535.5 |       477.7 |                317.5 |                -260.4 |
| Linear Regression    |        72 | WP      |         631 |     522.7 |       481.7 |                343.7 |                -223.5 |
| Linear Regression    |        72 | EP      |         493 |     428.7 |       382.2 |                241.8 |                -222.8 |
| Linear Regression    |        72 | NA      |         822 |     541.2 |       484.7 |                335.4 |                -248.8 |
| Random Forest        |        72 | SI      |         650 |     497.4 |       437.3 |                286.4 |                -268.3 |
| Random Forest        |        72 | WP      |         631 |     514.1 |       467.3 |                348.7 |                -166.9 |
| Random Forest        |        72 | EP      |         493 |     394.3 |       339.8 |                230.7 |                -177.5 |
| Random Forest        |        72 | NA      |         822 |     515.2 |       463.8 |                308.3 |                -286.3 |
| XGBoost              |        72 | SI      |         650 |     495.9 |       446.8 |                268.5 |                -294   |
| XGBoost              |        72 | WP      |         631 |     501   |       463.2 |                316   |                -207.2 |
| XGBoost              |        72 | EP      |         493 |     402.2 |       360.7 |                203.3 |                -235   |
| XGBoost              |        72 | NA      |         822 |     519.1 |       467.9 |                299.6 |                -305.5 |

## Reading the error decomposition

- **Cross-track error** is perpendicular to the storm's actual motion: a direction error, typically a missed or mistimed recurvature.
- **Along-track error** is parallel to motion: a speed error. The MAE is its magnitude; the bias is its systematic component. A bias close to the MAE means the model is consistently wrong in one direction (negative = it runs storms too slow) rather than merely imprecise, which is a correctable error.
