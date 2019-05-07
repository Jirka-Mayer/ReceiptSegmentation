# Dataset

Data pro testování systému jsou ve složce `/dataset`.

Ve složce `/dataset/img` jsou fotografie účtenek.

Soubor `/dataset/dataset.json` popisuje dataset a je v něm pole objektů, kde každý objekt reprezentuje jednu fotografii.

```json
{
    "img": "000.jpg",
    "difficulty": "hard",
    "quad": [
        [144, 406],
        [1038, 368],
        [1071, 775],
        [110, 779]
    ],
    "distribution": [
        [400, 476],
        [725, 615]
    ]
}
```

- `img` je název odpovídajícího souboru ve složce fotografií
- `difficulty` je náročnost fotografie a je to jedna z hodnot:
    - `easy` by se měla detekovat dobře
    - `medium` detekce může selhat, protože účtenka sice dobře kontrastuje, ale leží přes ní předměty
    - `hard` účtenka nejspíš detekována nebude, splývá s pozadím
- `quad` jsou souřadnice rohů čtyřúhelníka ve formátu `[x, y]`, první vlevo nahoře a potom ve směru hodinových ručiček další
- `distribution` je obdélníková oblast někde uvnitř účtenky, která by měla mít cílovou distribuci pixelů a je to obdélník zadaný levým horním rohem a pravým dolním rohem
