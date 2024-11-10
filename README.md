# tifs-type2branch
Companion source code for the article Type2Branch: Keystroke Biometrics based on a Dual-branch Architecture with Attention Mechanisms and Set2set Loss, to be published in IEEE Transactions on Information Forensics and Security.


Abstract

In 2021, the pioneering work on TypeNet showed that keystroke dynamics verification could scale to hundreds of thousands of users with minimal performance degradation. Recently, the KVC-onGoing competition has provided an open and robust experimental protocol for evaluating keystroke dynamics verification systems of such scale, including considerations of algorithmic fairness. This article describes Type2Branch, the model and techniques that achieved the lowest error rates at the KVC-onGoing, in both desktop and mobile scenarios. The novelty aspects of the proposed Type2Branch include: i) synthesized timing features emphasizing user behavior deviation from the general population, ii) a dual-branch architecture combining recurrent and convolutional paths with various attention mechanisms, iii) a new loss function named Set2set that captures the global structure of the embedding space, and iv) a training curriculum of increasing difficulty. Considering five enrollment samples per subject of approximately 50 characters typed, the proposed Type2Branch achieves state-of-the-art performance with mean per-subject EERs of 0.77% and 1.03% on evaluation sets of respectively 15,000 and 5,000 subjects for desktop and mobile scenarios. With a uniform global threshold for all subjects, the EERs are 3.25% for desktop and 3.61% for mobile, outperforming previous approaches by a significant margin. 


Links

Article Preprint
https://arxiv.org/abs/2405.01088

IEEE BigData 2023 Keystroke Verification Challenge (KVC)
https://arxiv.org/pdf/2401.16559
