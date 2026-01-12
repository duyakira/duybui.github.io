import queue
import threading
import torch
import gc
import os
import cv2
import traceback
task_queue2 = queue.Queue()

def worker2():
    while True:
        func, args = task_queue2.get()
        input_path, output_path = args
        try:
            sr = func(input_path)
            cv2.imwrite(output_path, sr)
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

            task_queue2.task_done()

threading.Thread(target=worker2, daemon=True).start()


