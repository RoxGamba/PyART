from PyART.catalogs.sacra import Waveform_SACRA
import matplotlib.pyplot as plt

datapath = '../eob_uber_alles/data/sacra'

plt.figure
for i in range(162):
    try:
        wave = Waveform_SACRA(ID=i, datapath=datapath, nu_rescale=True, cut_final=200)
        print(f"#{i:03d} : {wave.metadata['name']}")
        plt.plot(wave.u, wave.hlm[(2,2)]['A'])
    except Exception as e:
        print(f"#{i:03d} failed : {e}")
plt.show()
        
