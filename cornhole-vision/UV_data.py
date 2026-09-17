import numpy as np

W, H = 1280, 720
Y_SIZE = W * H
UV_W, UV_H = W // 2, H // 2
UV_SIZE = UV_W * UV_H

with open('frame.yuv420', 'rb') as f:
    data = f.read()

Y = np.frombuffer(data[0:Y_SIZE], dtype=np.uint8).reshape(H, W)
U = np.frombuffer(data[Y_SIZE:Y_SIZE+UV_SIZE], dtype=np.uint8).reshape(UV_H, UV_W)
V = np.frombuffer(data[Y_SIZE+UV_SIZE:Y_SIZE+2*UV_SIZE], dtype=np.uint8).reshape(UV_H, UV_W)

def sample(label, x, y, patch=7):
    cx, cy = x // 2, y // 2          # chroma coords are half-res
    h = patch // 2
    u_patch = U[cy-h:cy+h+1, cx-h:cx+h+1]
    v_patch = V[cy-h:cy+h+1, cx-h:cx+h+1]
    print(f"{label}: U mean={u_patch.mean():.1f} std={u_patch.std():.1f}  "
          f"V mean={v_patch.mean():.1f} std={v_patch.std():.1f}")

# fill in coords picked from the PNG (full-res, Y-plane coordinates)
sample("red bag",   x_red,   y_red)
sample("blue bag",  x_blue,  y_blue)
sample("board bg",  x_board, y_board)
