# Improved Interpretability and Generalisation for Deep Learning

Generalisation and interpretability of Recurrent Neural Networks with gated units are extremely challenging tasks. In this work the Stimulated Learning framework [Wu, Chunyang, et al., 2016] is applied to LSTMs for language modelling. 

The output activations of the LSTM cells are reorganised into a 2D flat grid, which is then converted into a toroidal 3D surface. Spatial smoothing is performed (both at the 2D and at the 3D level) to improve generalisation. Word2vec embeddings are then used in the 2D space to investigate interpretability. 

<p align="center">
    <img src="examples/torus.png" width=700px>
</p>
<p align="center">
    <img src="examples/gen.png" width=700px>
</p>

The preliminary results of this work show that spatial smoothness improves generalisation.


Related Work: 

Wu, Chunyang, et al. Stimulated deep neural network for speech recognition. University of Cambridge Cambridge, 2016. (<http://mi.eng.cam.ac.uk/~mjfg/interspeech16_stimu.pdf>)

Ragni, A., et al. "Stimulated training for automatic speech recognition and keyword search in limited resource conditions." Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. IEEE, 2017.(<http://mi.eng.cam.ac.uk/~mjfg/CUED-Ragni-Stimulated-ASR-KWS.pdf>)
 
 MPhil thesis: 

Improved Interpretability and Generalization for Deep Learning (<http://www.mlsalt.eng.cam.ac.uk/foswiki/pub/Main/ClassOf2017/Graziani_dissertation.pdf>)
