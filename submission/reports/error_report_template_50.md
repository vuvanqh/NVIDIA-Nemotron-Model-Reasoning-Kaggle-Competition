# Local Proxy Error Report

This report is based on the local proxy metric only. It is not the official competition metric.

- Scored input: `submission/data/scored_template_50.jsonl`
- Total examples: 50
- Correct: 0
- Incorrect: 50
- Accuracy: 0.000000

## Error Type Counts

Counts are non-exclusive; one incorrect row can contribute to multiple categories.

| Error type | Count |
| --- | ---: |
| missing prediction | 50 |
| no boxed answer | 50 |
| numeric fallback used | 0 |
| exact mismatch | 0 |
| numeric mismatch | 0 |

## Errors By Task Type

| Task type | Examples | Correct | Incorrect | Accuracy | Missing prediction | No boxed answer | Numeric fallback used | Exact mismatch | Numeric mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bit Manipulation | 7 | 0 | 7 | 0.000000 | 7 | 7 | 0 | 0 | 0 |
| Equations & Symbolic | 3 | 0 | 3 | 0.000000 | 3 | 3 | 0 | 0 | 0 |
| Numeral Conversion | 10 | 0 | 10 | 0.000000 | 10 | 10 | 0 | 0 | 0 |
| Physics Gravity | 19 | 0 | 19 | 0.000000 | 19 | 19 | 0 | 0 | 0 |
| Text Encryption | 5 | 0 | 5 | 0.000000 | 5 | 5 | 0 | 0 | 0 |
| Unit Conversion | 6 | 0 | 6 | 0.000000 | 6 | 6 | 0 | 0 | 0 |

## First Incorrect Examples

- `0dcdd12b` `Physics Gravity`: missing prediction | target=`38.99` | extracted=``
- `a897b8bc` `Bit Manipulation`: missing prediction | target=`01111101` | extracted=``
- `0adca57b` `Numeral Conversion`: missing prediction | target=`V` | extracted=``
- `ec1ee4b2` `Physics Gravity`: missing prediction | target=`95.02` | extracted=``
- `1db21126` `Physics Gravity`: missing prediction | target=`96.57` | extracted=``
- `d662a4c4` `Unit Conversion`: missing prediction | target=`14.15` | extracted=``
- `7e3fefc6` `Equations & Symbolic`: missing prediction | target=`?` | extracted=``
- `e24b1277` `Bit Manipulation`: missing prediction | target=`11101111` | extracted=``
- `abfd833f` `Unit Conversion`: missing prediction | target=`57.14` | extracted=``
- `b9623afc` `Numeral Conversion`: missing prediction | target=`XXXV` | extracted=``
