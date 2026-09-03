# Modelli Acustici (.npz)

In questa cartella vengono salvati i modelli acustici leggeri (`.npz`) addestrati sulle feature deep YAMNet (vettori di embedding a 1024 dimensioni) tramite AutoCut.

I modelli `.npz` pesano pochi megabyte e possono essere riutilizzati su qualsiasi video senza dover eseguire TensorFlow o rieseguire il training da zero.

## Come Salvare un Modello

Quando alleni con `--teach` o `--train-from`, salva il modello nella cartella `models/`:

```bash
python autocut.py video.mp4 --teach "1:15-1:18,2:04-2:07" --save-model models/mio_suono.npz
```

I nuovi modelli salvati con lo stesso nome accumulano automaticamente gli esempi precedenti (Experience Replay continuo).

## Come Riutilizzare un Modello Salvato

Puoi passare il nome del modello direttamente con `--load-model`:

```bash
# Cerca automaticamente sia il percorso esatto che la cartella models/
python autocut.py video.mp4 --load-model mio_suono.npz
```

Oppure specificando il percorso completo:

```bash
python autocut.py video.mp4 --load-model models/mio_suono.npz
```
