"""Strong general net (zhizi b40) on the classical boards."""
from katago import KataGoBot, net

BOT = KataGoBot("katago-zhizi-b40", "KataGo zhizi b40",
                model=net("katago_zhizi-b40.bin.gz"))
