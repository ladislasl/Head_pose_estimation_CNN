
from argparse import ArgumentParser

import cv2

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator



# Parser afin de donner l'input depuis une ligne de commande, par default ce sera la camera.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
args = parser.parse_args()


if __name__ == '__main__':

    # 1. Source = input de la ligne de commande donné par le parser.
    video_src = args.video
    if video_src is None:
        print("Video source not assigned")
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    # on récupère la taille d'une frame de la vidéo (une image)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 2. On instancie les classes utiles à savoir ici le pose estimator qui reçoit les landmark et en déduit la pose du visage !
    pose_estimator = PoseEstimator(img_size=(height, width))

    # 3. mark detector détecte les "landmarks" (traits caractéristiques du visages)
    mark_detector = MarkDetector()

    # 4. mesure de la performance
    tm = cv2.TickMeter()

    # on étudie maintenant frame par frame la video
    while True:

        # on ouvre et on lie la frame du moment dans la boucle
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Le réseau de neurones convolutionnel extrait un visage de la frame (modèle pré entrainé)
        facebox = mark_detector.extract_cnn_facebox(frame)

        # si on a trouvé un visage alors: 
        if facebox is not None:

            # Step 2: on extrait les "landmarks"
            
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            
            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

           
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Maintenant que on a les landmarks on essaye de trouver une pose du visage
            pose = pose_estimator.solve_pose_by_68_points(marks)

            

            # on affiche le cube autour du visage
            pose_estimator.draw_annotation_box(
                frame, pose[0], pose[1], color=(0, 255, 0))

            # on affiche les axes du visage
            pose_estimator.draw_axes(frame, pose[0], pose[1])

            # on affiche les landmarks
            mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

        

        # on affiche la frame
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break
