import mlx.core as mx
import time

mx.set_default_device(mx.Device(mx.gpu, 0))
print("Starting mlx ops")
for i in range(10):
    a = mx.random.uniform(shape=(1000, 1000))
    b = mx.matmul(a, a)
    mx.eval(b)
    print(f"Iter {i} done")
    time.sleep(0.5)

print("Done")
