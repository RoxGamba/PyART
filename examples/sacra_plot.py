from PyART.catalogs.sacra import Waveform_SACRA
import matplotlib.pyplot as plt

path = '../eob_uber_alles/data/sacra'

low_quality = list(range(90,162))

len_min    = 800
#long_sims  = []
sims2print = []
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
        #len_sim = wave.u[-1]-wave.u[0]
        #if len_sim>=len_min:
        #    long_sims.append(i)
        if wave.metadata['chi1z']<0.7:
            sims2print.append(i)
    except Exception as e:
        print(f"#{i:03d} failed : {e}")
for i in range(1,3):
    plt.subplot(2,1,i)
    plt.legend(ncol=10, fontsize=6)

print('[', end='')
for i, sim in enumerate(sims2print):
    print(sim, end='')
    if i!=len(sims2print)-1:
        if (i+1)%10>0:
            print(', ', end='')
        else:
            print(', \\')
print(']')
plt.show()



#for i in range(1,10):
#    wave = Waveform_SACRA(ID=i, path=path, nu_rescale=True, cut_final=200)
#    plt.figure
#    plt.plot(wave.u, wave.hlm[(2,2)]['A'], label=str(i))
#    plt.legend()
#    plt.show()
