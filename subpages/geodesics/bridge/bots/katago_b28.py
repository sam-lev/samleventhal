"""Strong general net (b28) on the classical boards."""
from katago import KataGoBot, net

BOT = KataGoBot("katago-b28", "KataGo b28",
                model=net("katago_b28.bin.gz"))
