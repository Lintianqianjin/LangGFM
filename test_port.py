import subprocess
import time
import socket

def wait_for_port(port, host="localhost", timeout=30):
    start_time = time.time()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((host, port))
                # 如果能连接上，说明端口已被监听
                return True
            except ConnectionRefusedError:
                pass

        if (time.time() - start_time) > timeout:
            return False

        time.sleep(1)

def main():
    server_process = subprocess.Popen(["bash", "start_server.sh"])
    
    if wait_for_port(8016, "localhost", 30):
        print("端口8018已被监听，执行后续任务...")
    else:
        print("等待端口监听超时，服务未启动。")
        server_process.kill()

if __name__ == "__main__":
    main()

