import logging

PFPLOGGER = logging.getLogger("polyfingerprints")
PFPLOGGER.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
PFPLOGGER.addHandler(handler)
