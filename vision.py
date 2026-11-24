from itertools import combinations
from tictactoe import find_best_move
import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks


camera_id = 1 # identifiant de la caméra
camera_width = 1920 # largeur souhaitée en pixels
fps = 30 # fréquence souhaitée
warmup_frames = 30*1  #nombre de frames ignorées au démarrage (pour stabilisation de la capture)
threshold_grid = 80
grid_side_mm = 700


# ==========================================================
# CAMERA
# ==========================================================



def initialize_camera(camera_id=0, width=1080, fps=30, warmup_frames=30):
    """
    Ouvre, configure et stabilise la caméra.

    Entrées :
        camera_id    : identifiant de la caméra
        width        : largeur souhaitée en pixels
        fps          : fréquence souhaitée
        warmup_frames: nombre de frames ignorées au démarrage

    Sortie :
        cap : objet cv2.VideoCapture prêt à être utilisé
    """

    cap = cv2.VideoCapture(camera_id)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    # Ignorer les premières images
    for _ in range(warmup_frames):
        ret, _ = cap.read()

        if not ret:
            cap.release()
            raise RuntimeError(
                "Erreur pendant l'initialisation de la caméra"
            )

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("Caméra initialisée")
    print(f"Résolution : {actual_width} x {actual_height}")
    print(f"FPS configurés : {actual_fps}")

    return cap


def initialize_video(video_path, warmup_frames=0):
    """
    Ouvre une vidéo enregistrée.

    Entrée :
        video_path : chemin vers le fichier vidéo
        warmup_frames : nombre d'images à ignorer au début

    Sortie :
        cap : objet cv2.VideoCapture
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la vidéo")

    for _ in range(warmup_frames):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Vidéo trop courte pour ignorer les premières images")

    return cap



# ==========================================================
# FONCTIONS D'INITIALISATION VISION
# ==========================================================

class GridNotDetectedError(RuntimeError):
    pass


def detect_grid(frame):
    """
    Détecte les droites avec Sobel et Hough.

    Entrée :
        frame : image caméra (BGR)

    Sortie :
        Liste des 16 points d'intersection visibles (x, y) en pixels.
    """

    # 1. Conversion en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Dérivées horizontale et verticale
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Conserver uniquement les dérivées positives
    dx_positive = np.maximum(dx, 0)
    dy_positive = np.maximum(dy, 0)

    # 3. Fusion des dérivées positives
    magnitude = cv2.magnitude(dx_positive, dy_positive)

    # 4. Normalisation entre 0 et 255
    magnitude_display = cv2.normalize(
        magnitude,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 5. Seuillage binaire
    _, binary = cv2.threshold(
        magnitude_display,
        threshold_grid,
        255,
        cv2.THRESH_BINARY
    )

    # 6. Calculer la transformée de Hough
    hough_space, angles, distances = hough_line(binary)
    # Détecter les pics principaux et supprimer les doublons
    _, peak_angles, peak_distances = hough_line_peaks(
        hough_space,
        angles,
        distances,
        min_distance=20,
        min_angle=15,
        threshold= 0.30,
        num_peaks=8
    )
    # Convertir l'image binaire en BGR pour les annotations en couleur.
    annotated_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    height, width = binary.shape[:2]

    # Dessiner les droites en rouge.
    for angle, distance in zip(peak_angles, peak_distances):

        # Calculer un point appartenant à la droite
        x0 = distance * np.cos(angle)
        y0 = distance * np.sin(angle)

        # Calculer le vecteur directeur de la droite
        dx = -np.sin(angle)
        dy = np.cos(angle)

        # Tracer une droite suffisamment longue pour couvrir l'image
        length = 2 * np.hypot(width, height)

        pt1 = (
            int(round(x0 + length * dx)),
            int(round(y0 + length * dy))
        )
        pt2 = (
            int(round(x0 - length * dx)),
            int(round(y0 - length * dy))
        )

        cv2.line(annotated_binary, pt1, pt2, (0, 0, 255), 2)

    # Chaque droite vérifie x * cos(theta) + y * sin(theta) = rho.
    grid_points = {}
    polar_lines = list(zip(peak_distances, peak_angles))
    for (rho1, theta1), (rho2, theta2) in combinations(polar_lines, 2):
        cos1, sin1 = np.cos(theta1), np.sin(theta1)
        cos2, sin2 = np.cos(theta2), np.sin(theta2)
        determinant = cos1 * sin2 - sin1 * cos2
        if abs(determinant) < 1e-8:
            continue

        x = (rho1 * sin2 - sin1 * rho2) / determinant
        y = (cos1 * rho2 - rho1 * cos2) / determinant
        pixel_x, pixel_y = round(x), round(y)
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            grid_points[(pixel_x, pixel_y)] = None

    for point in grid_points:
        cv2.circle(annotated_binary, point, 5, (0, 255, 0), -1)

    # 10. Affichage des résultats
    cv2.imshow("Gradient total", magnitude_display)
    cv2.imshow("Image binaire", binary)
    cv2.imshow("Droites et intersections", annotated_binary)

    if len(grid_points) != 16:
        raise GridNotDetectedError(
            f"Grille non détectée ({len(grid_points)} intersections sur 16)"
        )

    return list(grid_points)


def compute_homography(grid_points, frame):
    """
    Transformer les pixels caméra en millimètres sur la grille.

    Entrée :
        grid_points : points détectés dans l'image
        frame       : image source (BGR)

    Sortie :
        H : matrice d'homographie 3x3
    """
    
    points = np.asarray(grid_points, dtype=np.float32)
    if len(points) < 4 or not np.isfinite(points).all():
        raise GridNotDetectedError("Coins de grille invalides")

    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    corners = cv2.approxPolyDP(
        hull, 0.02 * perimeter, True
    ).reshape(-1, 2)

    if len(corners) != 4 or not cv2.isContourConvex(corners):
        raise GridNotDetectedError("grille non detect")

    center = corners.mean(axis=0)
    angles = np.arctan2(
        corners[:, 1] - center[1],
        corners[:, 0] - center[0]
    )
    corners = corners[np.argsort(angles)]

    # Ordre horaire depuis le coin supérieur gauche dans l'image.
    corners = np.roll(corners, -int(np.argmin(corners.sum(axis=1))), axis=0)

    # Définir un carré de destination de 700 x 700 mm à 1 px/mm
    side = grid_side_mm
    square = np.array([
        [0, 0],
        [side, 0],
        [side, side],
        [0, side]
    ], dtype=np.float32)

    # Calculer l'homographie
    H = cv2.getPerspectiveTransform(corners, square)
    if not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3:
        raise GridNotDetectedError("Homographie de grille invalide")

    # Vérifier que la projection reste définie sur toute l'image caméra.
    height, width = frame.shape[:2]
    frame_corners = np.array(
        [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)],
        dtype=np.float64
    )
    denominators = frame_corners @ H[2, :2] + H[2, 2]
    if denominators.min() <= 0 <= denominators.max():
        raise GridNotDetectedError("grille non detect")

    return H


def detect_shape_in_cell(cell):
    """
    Détecter une croix ou un carré dans une case.

    Sorties :
        shape    : "cross", "square" ou None
        vertices : coordonnées Nx2 des sommets dans la case
    """

    # Convertir la case en niveaux de gris
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Vérifier que le contraste est suffisant
    dark, light = np.percentile(gray, (5, 95))
    if light - dark < 25:
        return None, None

    # Extraire les formes sombres sur fond clair
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Détecter les contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    height, width = gray.shape

    # Examiner les contours du plus grand au plus petit
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Ignorer les contours trop petits
        if area < 0.015 * width * height:
            break

        # Ignorer les contours touchant les bords
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue

        # Approximer le contour par un polygone
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(
            contour, 0.025 * perimeter, True
        )

        # Vérifier que la forme n'est pas trop allongée
        _, (side_a, side_b), _ = cv2.minAreaRect(contour)

        if min(side_a, side_b) < 0.65 * max(side_a, side_b):
            continue

        # Identifier un carré
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            shape = "square"

        # Identifier une croix
        elif (
            len(polygon) >= 8
            and area < 0.85 * cv2.contourArea(cv2.convexHull(contour))
        ):
            shape = "cross"

        else:
            continue

        # Renvoyer les sommets du polygone détecté
        vertices = polygon[:, 0].astype(np.float32)

        return shape, vertices

    return None, None


def detect_board(frame, H):
    """
    Détecter les formes dans la grille de 700 x 700 mm.

    Sorties :
        board    : matrice 3x3 ("cross", "square" ou None)
        vertices : matrice Nx2 des sommets (X, Y) en mm

    Repère fixe :
        Origine : coin supérieur gauche
        X : vers la droite
        Y : vers le bas
    """

    margin = 8

    # Inclure les deux bords à 0 et 700, à raison d'un pixel par mm.
    rectified = cv2.warpPerspective(
        frame, H, (grid_side_mm + 1, grid_side_mm + 1)
    )
    cv2.imshow("Image corrigee", rectified)

    # Initialiser la matrice des formes et la liste des sommets
    board = np.full((3, 3), None, dtype=object)
    all_vertices = []

    # Calculer les limites des neuf cases
    boundaries = np.rint(
        np.linspace(0, grid_side_mm, 4)
    ).astype(int)

    # Parcourir les trois lignes et les trois colonnes
    for row in range(3):
        for col in range(3):

            x0 = boundaries[col] + margin
            x1 = boundaries[col + 1] - margin
            y0 = boundaries[row] + margin
            y1 = boundaries[row + 1] - margin

            # Extraire la case en excluant les lignes de la grille
            cell = rectified[y0:y1, x0:x1]

            # Détecter la forme et ses sommets
            shape, vertices = detect_shape_in_cell(cell)

            board[row, col] = shape

            if vertices is not None:
                all_vertices.append(vertices + np.float32([x0, y0])) # Hema change extend in append to group obstacles point together 
    
    # # Convertir la liste en matrice Nx2
    # all_vertices = np.asarray(
    #     all_vertices, dtype=np.float32
    # ).reshape(-1, 2)

    for i, v in enumerate(all_vertices):
        print(f"obstacle {i}: {len(v)} vertices")
        print(np.round(v, 1))

    return board, all_vertices



def get_goal_position(best_move, H, frame, vertices_mm):
    """Afficher la cible et renvoyer (X, Y) en mm depuis le coin supérieur gauche."""
    annotated = frame.copy()
    to_camera = np.linalg.inv(H)
    if len(vertices_mm):
        camera_vertices = cv2.perspectiveTransform(
            vertices_mm[None, :, :], to_camera
        )[0]
        for point in np.rint(camera_vertices).astype(int):
            cv2.circle(annotated, tuple(point), 4, (0, 255, 0), -1)

    goal_mm = None
    if best_move is not None:
        row, col = best_move
        cell_size = grid_side_mm / 3
        goal_mm = ((col + 0.5) * cell_size, (row + 0.5) * cell_size)
        center_camera = cv2.perspectiveTransform(
            np.float32([[goal_mm]]), to_camera
        )[0, 0]
        cv2.drawMarker(
            annotated, tuple(np.rint(center_camera).astype(int)), (0, 0, 255),
            cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA
        )

    cv2.imshow("Cible et sommets", annotated)
    return goal_mm


# ==========================================================
# INITIALISATION COMPLETE DE LA VISION
# ==========================================================

def initialize_vision(frame):
    """
    Effectuer l'initialisation complète sur une seule image.

    Entrée :
        frame : image caméra

    Sortie :
        H : homographie vers la grille redressée
    """
    grid_points = detect_grid(frame)
    H = compute_homography(grid_points, frame)

    board, vertices_mm = detect_board(frame, H)
    best_move = find_best_move(board)
    goal_mm = get_goal_position(best_move, H, frame, vertices_mm)

    return H, vertices_mm, goal_mm


# ==========================================================
# TEMPS REEL
# ==========================================================

THYMIO_MARKER_ID = 0


def detect_thymio(frame, H, detector=None):
    """Afficher le suivi et renvoyer (x, y, theta) en mm et radians, ou None."""
    if detector is None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary)

    annotated = frame.copy()
    corners, ids, _ = detector.detectMarkers(frame)
    points = None
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.ravel()):
            if marker_id == THYMIO_MARKER_ID:
                points = marker_corners.reshape(4, 2)
                break
    thymio_pos = None
    label = "Thymio non detecte"
    if points is not None:
        points_mm = cv2.perspectiveTransform(points[None, :, :], H)[0]
        x, y = points_mm.mean(axis=0)
        dx, dy = points_mm[1] - points_mm[0]
        theta = np.arctan2(dy, dx)
        thymio_pos = float(x), float(y), float(theta)

        arrow_mm = np.float32([[[x, y], [x + 50 * np.cos(theta), y + 50 * np.sin(theta)]]])
        center, end = np.rint(
            cv2.perspectiveTransform(arrow_mm, np.linalg.inv(H))[0]
        ).astype(int)
        cv2.circle(annotated, tuple(center), 5, (0, 0, 255), -1)
        cv2.arrowedLine(annotated, tuple(center), tuple(end), (255, 0, 0), 3)
        label = f"Thymio ({x:.1f}, {y:.1f}) mm | {np.degrees(theta):.1f} deg"

    cv2.putText(annotated, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Thymio - Vision", annotated)
    return thymio_pos


# ==========================================================
# MAIN
# ==========================================================

def main():

    # --------------------------------------------------
    # LANCEMENT CAMERA
    # --------------------------------------------------

    cap = initialize_camera(camera_id, camera_width, fps, warmup_frames)

    try:
        # --------------------------------------------------
        # INITIALISATION VISION (UNE FOIS)
        # --------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Impossible de lire l'image")

        H, vertices_mm, goal_mm = initialize_vision(frame)
        print(H, vertices_mm, goal_mm)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(dictionary)

        # --------------------------------------------------
        # BOUCLE TEMPS REEL
        # --------------------------------------------------

        while True:

            thymio_pos = detect_thymio(frame, H, detector)
            print(thymio_pos)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                try:
                    H, vertices_mm, goal_mm = initialize_vision(frame)
                    print("Vision reinitialisee")
                except GridNotDetectedError as error:
                    print(error, flush=True)

            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Impossible de lire l'image de la caméra")

    except GridNotDetectedError as error:
        print(error, flush=True)
        cv2.waitKey(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
