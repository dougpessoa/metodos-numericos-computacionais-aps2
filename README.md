# APS 2 - Métodos Numéricos Computacionais
Nome: Douglas Pereira Pessoa \
Matrícula: 2020101810 \
Turma: 131 \
Email: 2020101810@aluno.unicarioca.edu.br \
Professor: Sérgio Assunção Monteiro


Rode no [Colab](https://colab.research.google.com/drive/1lw3aAn5pndJK7nehb0HbNKcYGKx1VTEj?usp=sharing)

___

### Rode as Importações

```python
import numpy as np
from numpy.linalg import svd
import cv2
import os
from skimage import io
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
```

### Questão (a)
```python
origem = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80"
image = io.imread(origem) 

img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(img)

cv2_imshow(img)
X = np.array(img)
```

### Questão (b)
```python
U, Singular, V = svd(X)
print("U: ",U)
print("Valores Singulares:", Singular)
print("V^{T}",V.T)
```

### Questão (c)

```python 
PERCENTAGE = 80
size = int(PERCENTAGE * img.shape[0] / 100)

rr = np.dot(U[:, :size], np.dot(np.diag(Singular[:size]), V[:size, :]))

print("rr",rr)
```

### Questão (d)

```python 
cv2_imshow(rr)
```