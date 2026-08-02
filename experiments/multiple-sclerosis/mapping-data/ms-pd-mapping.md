## **Disease Key Mapping**

**MS**: Multiple Sclerosis  
**PD**: Parkinson’s Disease  
**Normal**: Normal

## **Disease Landmarks**

| Disease | Features | MS Effect | Why we want it | Pose landmarks |
| ----- | ----- | ----- | ----- | ----- |
| **MS, PD** | **walking\_speed\_ms** | **Lower** | The single strongest overall indicator of MS and PD gait impairment. Reflects weakness, fatigue, and overall disability. | Left Hip (23), Right Hip (24) |
| **MS, PD** | **stride\_length\_m** | **Shorter** | One of the earliest and most consistent gait abnormalities. Indicates reduced propulsion and cautious gait. | Left Heel (29), Right Heel (30), Left Foot Index (31), Right Foot Index (32) |
| **MS, PD** | **double\_support\_percentage** | **Higher** | Patients keep both feet on the ground longer because of poor balance and fear of falling. One of the best balance metrics. | Heels (29,30), Foot Index (31,32) |
| **MS, PD** | **stride\_time\_cv** | **Higher variability** | Captures inconsistent walking. Strongly associated with fatigue, neurological impairment, fall risk. | Feet (29–32) |
| **MS** | **left/right\_knee\_range** | **Smaller ROM (less knee flexion)** | Measures stiff-legged gait caused by spasticity and weakness during swing. | Hip → Knee → Ankle (23-25-27, 24-26-28) |
| **MS** | **left/right\_ankle\_range**\* | **Smaller ROM** *(reduced dorsiflexion & plantarflexion)* | Detects foot drop and weak push-off. Ideally this would be replaced with separate dorsiflexion and plantarflexion features if we later compute gait phases. | Knee (25,26), Ankle (27,28), Heel (29,30), Foot Index (31,32) |
| **MS** | **hip\_asymmetry** | **Higher asymmetry** | MS often affects one side more than the other \-\> asymmetric gait especially in the hips. | Left Hip (23), Right Hip (24) |
| **PD** | **shoulder\_symmetry\_index** | **Lower symmetry (Higher asymmetry** | PD often produces asymmetric gait in the upper body, typically reduced arm swing, which is more pronounced on one side. | Left shoulder (11), right shoulder (12) |
| **PD** | **trunk\_lean\_angle** | **Increased forward trunk lean** | Common trait in PD caused by rigidity and muscle stiffness, leads to persistent forward stoop. | Left Shoulder (11), Right Shoulder (12), Left Hip (23), Right Hip (24) |
| **MS, PD** | **step\_width\_m** | **MS: wider, PD: narrower** | PD patients typically walk with a smaller base of support, opposite direction from the wide-based ataxic gait seen in MS.  | Left Ankle (27), Right Ankle (28) |

## **List of important pose landmarks**

left\_shoulder (11),  
right\_shoulder (12),

left\_hip (23),  
left\_knee (25),  
left\_ankle (27),  
left\_heel (29),  
left\_foot\_index (31),

right\_hip (24),  
right\_knee (26),  
right\_ankle (28),  
right\_heel (30),  
right\_foot\_index (32)

## **Differences between MS and PD**

| Dimension | Multiple Sclerosis (MS) | Parkinson's Disease (PD) |
| ----- | ----- | ----- |
| Base of support (step width) | Wider, more variable. Ataxic/unsteady | Narrower than healthy controls |
| Best lab discriminator | Toe-off angle, foot push-off weakness | Lumbar coronal ROM |
| Step size character | Shortened, but inconsistent step-to-step (weakness \+ cautious gait). | Shortened and *rhythmic/shuffled*, sometimes progressively shrinking (festination) |
| Trunk/arm | Not a primary sign | Reduced arm swing, forward-stooped posture, often asymmetric |
| Freezing | Not characteristic | \~Half of PD patients experience freezing of gait |

## **Compilation of research on MS**

The next few features outlined below will hopefully serve as a guide for the specific features used for our model. From various studies comparing MS gait with healthy gait, it has been established that MS gait has reduced speed (MS 1.2 m/s vs control 1.42 m/s, Kelleher et al. 2010\) and reduced step length (MS 45.3 cm vs control 72.1 cm, Givon et al. 2009). During the stance period, during which at least one part of the foot is in contact with the ground, the amount of hip extension, which is the backward movement of the thigh, decreases (MS 0.650 Nm/Kg vs control 0.789 Nm/Kg, Huisinga et al. 2013). In the swing period, when the foot is off the ground, knee flexion (bending the knee when bringing the heel backwards) decreases (MS 35º vs control 55º, Filli et al. 2018). During initial contact, the moment the foot strikes the ground with the heel, ankle dorsiflexion (bringing the front of the foot up) decreases (MS: \-10º vs control \-5º, FIlli et al. 2018). In the pre-swing phase (the transition between the stance and swing phase), ankle plantar flexion (the downward movement of the foot when the foot pushes off the ground) decreases (MS \-14.67º vs control \-20º, Morel et al. 2017).

| MS sign | How they are manifested on video | What our program could measure |
| :---- | :---- | :---- |
| Weakness | Slow push-off or dragging foot | Shorter stride length/reduced ankle power |
| Spasticity | Stiff-legged swing or less fluid motion | Reduced knee flexion in swing/ankle dorsiflexion |
| Balance problem | Wobbling, cautious steps, or instability in turns | More double-support time, higher variability |
| Foot placement  | Toe drag, heel strike problems, or tripping | Reduced dorsiflexion |
| Fatigue  | Slower walking and more variability over time | Speed decreasing and higher variability |

* Compared to Parkinson’s disease, which have easily identifiable walking patterns, MS walking patterns are more variable.  
* For a gait analysis video, the clearest visible features are: reduced gait speed, shorter stride length, hesitation, foot placement, and tripping/stumbling. In a more detailed analysis, MS can also show reduced knee flexion during swing, reduced ankle dorsiflexion at initial contact, and reduced ankle plantar flexion during pre-swing.

