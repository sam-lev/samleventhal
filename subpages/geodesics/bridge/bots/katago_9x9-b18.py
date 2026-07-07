"""9x9-specialist net — gated to the 9x9 plane board only (scaleIdx 0)."""
from katago import KataGoBot, net

BOT = KataGoBot("katago-9x9", "KataGo 9\u00d79",
                model=net("katago_9x9-b18.bin.gz"),
                supports={"surfaces": ["plane"], "meshes": ["square"],
                          "incidence": ["vertices"], "scaleIdx": [0]})
