# Solar Panel Datasets

##

| 구분 | 설명 | 카테고리 |
| --- | --- | ------ |
| PVEL-AD | Solar cell EL image defect detection dataset <br> [github.com/binyisu/PVEL-AD](https://github.com/binyisu/PVEL-AD) | 태양광 점검 / 데이터셋 |
| InfraredSolarModules | 20,000 infrared images (24 x 40 pixels) <br> 12 defined classes of solar modules <br> [github.com/RaptorMaps/InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) | 태양광 점검 / 데이터셋 (적외선) |
| PV-Multi-Defect | PV paneldefect detection <br> [github.com/CCNUZFW/PV-Multi-Defect](https://github.com/CCNUZFW/PV-Multi-Defect) | 태양광 점검 / 데이터셋 |
| elpv-dataset | Visual Identification of Defective Solar Cells in Electroluminescence Imagery <br> [github.com/zae-bayern/elpv-dataset](https://github.com/zae-bayern/elpv-dataset) | 태양광 점검 / 데이터셋 |


####

a dataset of solar cell images extracted from high-resolution electroluminescence images of photovoltaic modules.

- 2,624 샘플 (300 x 300 px, 8-bit 회색 이미지)

####

a machine learning dataset that contains real-world imagery of different anomalies

- 20,000 적외선 이미지 (24 x 40 px)

|   클래스         | 이미지수 |                                설명                                        |
|:--------------:|:------:|:-------------------------------------------------------------------------:|
| Cell           | 1,877  | Hot spot occurring with square geometry in single cell.                   |
| Cell-Multi     | 1,288  | Hot spots occurring with square geometry in multiple cells.               |
| Cracking       | 941    | Module anomaly caused by cracking on module surface.                      |
| Hot-Spot       | 251    | Hot spot on a thin film module.                                           |
| Hot-Spot-Multi | 247    | Multiple hot spots on a thin film module.                                 |
| Shadowing      | 1056   | Sunlight obstructed by vegetation, man-made structures, or adjacent rows. |
| Diode          | 1,499  | Activated bypass diode, typically 1/3 of module.                          |
| Diode-Multi    | 175    | Multiple activated bypass diodes, typically affecting 2/3 of module.      |
| Vegetation     | 1,639  | Panels blocked by vegetation.                                             |
| Soiling        | 205    | Dirt, dust, or other debris on surface of module.                         |
| Offline-Module | 828    | Entire module is heated.                                                  |
| No-Anomaly     | 10,000 | Nominal solar module.                                                     |