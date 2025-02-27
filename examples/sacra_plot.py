from PyART.catalogs.sacra import Waveform_SACRA
import matplotlib.pyplot as plt

path = '../eob_uber_alles/data/sacra'

low_quality = list(range(90,162))

plt.figure(figsize=(12,8))
for i in range(1,162):
    try:
        wave = Waveform_SACRA(ID=i, path=path, nu_rescale=True, cut_final=200)
        print(f"#{i:03d} : {wave.metadata['name']}")
        if i in low_quality:
            plt.subplot(2,1,2)
        else:
            plt.subplot(2,1,1)
        plt.plot(wave.u, wave.hlm[(2,2)]['A'], label=str(i))
    except Exception as e:
        print(f"#{i:03d} failed : {e}")
for i in range(1,3):
    plt.subplot(2,1,i)
    plt.legend(ncol=10, fontsize=6)
plt.show()
 

#for i in range(1,10):
#    wave = Waveform_SACRA(ID=i, path=path, nu_rescale=True, cut_final=200)
#    plt.figure
#    plt.plot(wave.u, wave.hlm[(2,2)]['A'], label=str(i))
#    plt.legend()
#    plt.show()
