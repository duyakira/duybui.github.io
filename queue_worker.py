import queue
import threading
import torch
import gc
import os
import traceback


task_queue = queue.Queue()


def worker():
    while True:
        func, args = task_queue.get()
        input_path, output_path = args

        try:
            func(*args)
            
            open(output_path + ".done", "w").close() #Đảm bảo có thể nhận được nhiều hơn 2 requests
            
        except Exception as e:
            print("❌ Worker error:")
            traceback.print_exc()

        finally:
            # ❗ chỉ xoá input nếu output đã tồn tại
            if os.path.exists(output_path) and os.path.exists(input_path):
                os.remove(input_path)

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            task_queue.task_done()

threading.Thread(target=worker, daemon=True).start()



