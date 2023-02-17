
import cv2
import time
import os


# 이미지에 텍스트를 출력하는 함수
def draw_text(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color=(255, 0, 0)
    text_color_bg=(0, 0, 0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    offset = 5

    cv2.rectangle(img, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)


# 웹캠 연결
cap = cv2.VideoCapture("http://192.168.0.199:8080/?action=stream")


# 웹캠에서 fps 값 획득
fps = cap.get(cv2.CAP_PROP_FPS)
print('fps', fps)

if fps == 0.0:
    fps = 30.0

time_per_frame_video = 1/fps
last_time = time.perf_counter()


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
writer = cv2.VideoWriter(BASE_DIR + '/' + 'output.mp4', fourcc, fps, (width, height))

while True:

    # 웹캠에서 이미지 읽어옴
    ret,img_color = cap.read()

    if ret == False:
        print('웹캠에서 영상을 읽을 수 없습니다.')
        break

    writer.write(img_color)

    # fsp 계산
    time_per_frame = time.perf_counter() - last_time
    time_sleep_frame = max(0, time_per_frame_video - time_per_frame)
    time.sleep(time_sleep_frame)

    real_fps = 1/(time.perf_counter()-last_time)
    last_time = time.perf_counter()


    x = 30
    y = 50
    text = '%.2f fps' % real_fps

    # 이미지의 (x, y)에 텍스트 출력
    draw_text(img_color, text, x, y)
    cv2.imshow("Color", img_color)


    # ESC키 누르면 중지
    if cv2.waitKey(1)&0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()