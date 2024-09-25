import logging
import os

from cityvision import launcher

if __name__ == "__main__":
    os.environ["GOOGLE_CLOUD_PROJECT"] = "apps-cityvision-prod"
    logging.getLogger().setLevel(logging.INFO)
    launcher.run()