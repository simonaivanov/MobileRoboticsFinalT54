import cv2


def main():
    camera = cv2.VideoCapture(0)
    window_name = "Aperçu de la caméra"

    try:
        if not camera.isOpened():
            raise SystemExit("Impossible d'ouvrir la caméra.")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit("Impossible de lire une image de la caméra.")

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
