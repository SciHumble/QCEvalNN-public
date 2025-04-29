from pushbullet import Pushbullet
import configparser
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
notification_file = Path(__file__).parent / "notification.ini"

try:
    config = configparser.ConfigParser()
    config.read(notification_file)
    api_key = config["pushbullet"]["api_key"]
except Exception as e:
    logger.info(f"There will be no 'Pushbullet' notifications because the API "
                f"key couldn't be read due to: {e!r}.")
    api_key = None


def notify(title, message):
    if api_key is None:
        logger.debug("No message was sent because the API key is missing.")
    else:
        try:
            pb = Pushbullet(api_key)
            response = pb.push_note(title, message)
            logger.debug(f"A message was sent with the response: {response}")
            logger.info(f"Notification sent: {title} - {message}")
        except Exception as e:
            logger.warning(f"Pushbullet did not work because of: {e!r}.")
            logger.info(f"Notification attempted: {title} - {message}")


if __name__ == "__main__":
    # Test notification
    notify("Test Notification", "This was sent via an API.")
