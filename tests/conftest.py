import sys
import types
from unittest.mock import MagicMock

# Create a dummy module for notifications with a disabled 'notify' function
dummy_notification = types.ModuleType("qc_eval.misc.notification")
dummy_notification.notify = MagicMock(return_value=None)

# Override the actual module in sys.modules so that any subsequent import of
# 'qc_eval.misc.notification' gets the dummy module
sys.modules["qc_eval.misc.notification"] = dummy_notification
