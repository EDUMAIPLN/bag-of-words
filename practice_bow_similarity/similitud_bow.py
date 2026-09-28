"""
Práctica 1 PLN: Similitud de documentos con Bag-of-Words y similitud del coseno.

Pasos:
  1. Preprocesamiento (minúsculas, eliminación de caracteres no alfabéticos,
     eliminación opcional de stop words).
  2. Construcción del vocabulario global y de los vectores de frecuencias.
  3. Cálculo de la similitud del coseno entre todos los pares de documentos.
  4. Análisis: pares más/menos similares y similitud media por tema.

Uso:  python similitud_bow.py [carpeta_con_docx]
Requiere: python-docx, numpy, matplotlib (solo para el mapa de calor).
"""

import sys
import re
import unicodedata
from pathlib import Path
from itertools import combinations

import numpy as np
from docx import Document

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
CARPETA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/uploads")

# Etiquetado manual del tema de cada documento (para el análisis)
TEMAS = {
    "doc01": "Móvil", "doc02": "Móvil", "doc03": "Móvil",
    "doc06": "Móvil", "doc07": "Móvil", "doc10": "Móvil",
    "doc04": "Eléctrico", "doc05": "Eléctrico",
    "doc08": "Eléctrico", "doc09": "Eléctrico",
    "doc11": "Ambos", "doc12": "Ambos",
}

# Lista de stop words en español (artículos, preposiciones, pronombres,
# conjunciones y verbos auxiliares muy frecuentes)
STOP_WORDS = set("""
a al algo algunas algunos ante antes aquellos aquel aquella como con contra
cual cuales cuando de del desde donde dos durante e el ella ellas ellos en
entre era es esa ese eso esta este esto estos estas están esta está estar
fue fueron ha han hasta hay la las le les lo los mas más me mi muy nada ni
no nos nuestra nuestras nuestro nuestros o otra otras otro otros para pero
poco por porque puede pueden que quien se sea ser si sí sin sobre solo su
sus también tan te tiene tienen todo todos tras tu un una uno unos unas y
ya cada vez según ha así además embargo aquellos les lleva
""".split())


def leer_docx(ruta: Path) -> str:
    """Devuelve todo el texto (título incluido) de un .docx."""
    return "\n".join(p.text for p in Document(ruta).paragraphs)


# ---------------------------------------------------------------------------
# 1. Preprocesamiento
# ---------------------------------------------------------------------------
def preprocesar(texto: str, quitar_stop_words: bool = True) -> list[str]:
    texto = texto.lower()
    # Normalizamos para comparar sin problemas de codificación, pero
    # conservamos tildes y ñ, que son letras válidas en español.
    texto = unicodedata.normalize("NFC", texto)
    # Todo lo que no sea letra (incluye dígitos y puntuación) -> espacio
    texto = re.sub(r"[^a-záéíóúüñ]+", " ", texto)
    tokens = texto.split()
    if quitar_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


# ---------------------------------------------------------------------------
# 2. Bag-of-Words
# ---------------------------------------------------------------------------
def construir_vocabulario(docs_tokens: list[list[str]]) -> list[str]:
    return sorted({t for tokens in docs_tokens for t in tokens})


def vectorizar(tokens: list[str], indice: dict[str, int]) -> np.ndarray:
    v = np.zeros(len(indice), dtype=float)
    for t in tokens:
        v[indice[t]] += 1
    return v


# ---------------------------------------------------------------------------
# 3. Similitud del coseno
# ---------------------------------------------------------------------------
def coseno(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def matriz_similitud(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    S = np.eye(n)
    for i, j in combinations(range(n), 2):
        S[i, j] = S[j, i] = coseno(X[i], X[j])
    return S


# ---------------------------------------------------------------------------
# 4. Análisis
# ---------------------------------------------------------------------------
def analizar(nombres, S, titulo):
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")

    # Matriz
    print("       " + " ".join(n[-2:].rjust(5) for n in nombres))
    for i, n in enumerate(nombres):
        print(n.ljust(7) + " ".join(f"{S[i, j]:5.2f}" for j in range(len(nombres))))

    pares = sorted(
        ((S[i, j], nombres[i], nombres[j]) for i, j in combinations(range(len(nombres)), 2)),
        reverse=True,
    )
    print("\nTop 5 pares MÁS similares:")
    for s, a, b in pares[:5]:
        print(f"  {a} ({TEMAS[a]:9}) - {b} ({TEMAS[b]:9}): {s:.3f}")
    print("Top 5 pares MENOS similares:")
    for s, a, b in pares[-5:]:
        print(f"  {a} ({TEMAS[a]:9}) - {b} ({TEMAS[b]:9}): {s:.3f}")

    # Similitud media por combinación de temas
    grupos: dict[tuple, list] = {}
    for s, a, b in pares:
        clave = tuple(sorted((TEMAS[a], TEMAS[b])))
        grupos.setdefault(clave, []).append(s)
    print("\nSimilitud media por combinación de temas:")
    for clave, vals in sorted(grupos.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {clave[0]:9} - {clave[1]:9}: {np.mean(vals):.3f}  (n={len(vals)})")
    return pares, grupos


def dibujar(nombres, S, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    etiquetas = [f"{n} ({TEMAS[n][0]})" for n in nombres]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(S, cmap="YlOrRd", vmin=0, vmax=max(0.4, S[~np.eye(len(S), dtype=bool)].max()))
    ax.set_xticks(range(len(nombres)), etiquetas, rotation=60, ha="right")
    ax.set_yticks(range(len(nombres)), etiquetas)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            if i != j:
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Similitud del coseno (BoW sin stop words)\nM = móvil, E = eléctrico, A = ambos")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)


def main():
    rutas = sorted(CARPETA.glob("doc*.docx"))
    nombres = [r.stem for r in rutas]
    textos = [leer_docx(r) for r in rutas]
    print(f"Documentos cargados: {len(textos)}")

    resultados = {}
    for quitar in (False, True):
        tokens = [preprocesar(t, quitar) for t in textos]
        vocab = construir_vocabulario(tokens)
        indice = {w: i for i, w in enumerate(vocab)}
        X = np.array([vectorizar(t, indice) for t in tokens])
        S = matriz_similitud(X)
        etiqueta = "SIN stop words" if quitar else "CON stop words"
        print(f"\nVocabulario {etiqueta}: {len(vocab)} palabras")
        resultados[quitar] = analizar(nombres, S, f"Similitud del coseno ({etiqueta})")

        if quitar:
            # Palabras compartidas que explican los pares más similares
            print("\nPalabras compartidas en los 3 pares más similares:")
            for s, a, b in resultados[quitar][0][:3]:
                i, j = nombres.index(a), nombres.index(b)
                comunes = [vocab[k] for k in np.nonzero(X[i] * X[j])[0]]
                print(f"  {a}-{b}: {', '.join(comunes)}")
            dibujar(nombres, S, Path(__file__).with_name("mapa_similitud.png"))


if __name__ == "__main__":
    main()
