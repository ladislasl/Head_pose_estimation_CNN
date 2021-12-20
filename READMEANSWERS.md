1°) A lot of methods can be used for head pose estimation and several are considered sota, I have seen that a new method called WHENet is getting great results and is beating classic sota methods for frontal head pose estimation. this is the abstract from the article i found:

We present an end-to-end head-pose estimation network designed to predict Euler angles through the full range head yaws from a single RGB image. Existing methods perform well for frontal views but few target head pose from all viewpoints. This has applications in autonomous driving and retail. Our network builds on multi-loss approaches with changes to loss functions and training strategies adapted to wide range estimation. Additionally, we extract ground truth labelings of anterior views from a current panoptic dataset for the first time. The resulting Wide Headpose Estimation Network (WHENet) is the first fine-grained modern method applicable to the full-range of head yaws (hence wide) yet also meets or beats state-of-the-art methods for frontal head pose estimation. Our network is compact and efficient for mobile devices and applications.


link to the article : https://arxiv.org/abs/2005.10353v2


2°) What i coded to find Which video sample match which vector sets can be found in the notebook called reponses.ipynb.
    -the file code_source contains the algorithms i coded to solve the problem, they are all presented in the notebook reponses.ipynb
    -the files vid0, vid1, vid2 contain all the frames i used to study the different videos
    




