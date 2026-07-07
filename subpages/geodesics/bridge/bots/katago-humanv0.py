"""Human SL net — imitates players of a chosen rank (policy sampling at
1 visit under a humanSLProfile, per KataGo's Human SL guide). If your
KataGo version reports an unknown profile string, adjust the table below;
existing profiles span preaz_20k ... preaz_9d and proyear_1800 ... 2023."""
from katago import KataGoBot, net

BOT = KataGoBot("katago-human", "KataGo human",
                model=net("katago-humanv0.bin.gz"),
                human_profiles={
                    "20 kyu": "preaz_20k",
                    "10 kyu": "preaz_10k",
                    "5 kyu":  "preaz_5k",
                    "1 kyu":  "preaz_1k",
                    "1 dan":  "preaz_1d",
                    "5 dan":  "preaz_5d",
                    "9 dan":  "preaz_9d",
                })
