"""
End-to-End 파이프라인 실행
===========================

전체 흐름:
    1. 자동 라벨링 (heuristic/sam2/yolo_world)
    2. 시각화 → 수동 검수
    3. (Label Studio로 수정 후 re-export)
    4. 데이터셋 분할 & data.yaml 생성
    5. YOLO11n 학습 시작

사용:
    # Round 0: 50장 수동 라벨링 후 초기 학습
    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_seed \
        --output ./workspace/round_0

    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_s10 \
        --device mps \
        --output ./workspace/round_s10

    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_s20 \
        --model models/yolo11x.pt \
        --device mps \
        --output ./workspace/round_s20

    # Round 1+: 자동 라벨 → 샘플 선별
    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_0/weights/weights/best.pt \
        --output ./workspace/round_1 \
        --select-top 20

    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_s10/weights-2/weights/best.pt \
        --device mps \
        --output ./workspace/round_1_s10 \
        --select-top 20

    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_s20/weights/weights/best.pt \
        --device mps \
        --output ./workspace/round_1_s20 \
        --select-top 20

    # Round 2: 100장 라벨링 보정 후 학습
    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_seed_r2 \
        --output ./workspace/round_2

    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_r2_50 \
        --device mps \
        --output ./workspace/round_2_50

    # Round 3+: 자동 라벨 → 샘플 선별
    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_2/weights/weights/best.pt \
        --output ./workspace/round_3 \
        --select-top 20

    # Round 2: 100장 라벨링 보정 후 학습
    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --device mps \
        --seed-labels ./workspace/labels_seed_r2 \
        --output ./workspace/round_2

    # Round 3+: 자동 라벨 → 샘플 선별
    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_2/weights5/weights/best.pt \
        --output ./workspace/round_3 \
        --select-top 20

    # Round 3+: 자동 라벨 → 샘플 선별
    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_2/weights-5/weights/best.pt \
        --device mps \
        --output ./workspace/round_3 \
        --select-top 20

    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_2_50/weights/weights/best.pt \
        --device mps \
        --output ./workspace/round_3_50 \
        --select-top 20

    # Round 4: 100장 라벨링 보정 후 학습 (Round 0: Seed 20장, Round 2: 50장, Round 4: 100장)
    # 100장으로 학습할 때 MPS에서 오류가 발생
    python scripts/run_active_training.py seed \
        --images data/solar/images/RGB \
        --seed-labels ./workspace/labels_r4_100 \
        --device cpu \
        --output ./workspace/round_4_100

    # Round 5+: 자동 라벨 → 샘플 선별
    python scripts/run_active_training.py iterate \
        --images data/solar/images/RGB \
        --model ./runs/detect/workspace/round_4_100/weights-3/weights/best.pt \
        --device mps \
        --output ./workspace/round_5 \
        --select-top 20


python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s20 \
    --model models/yolo11s.pt \
    --device mps \
    --output ./workspace/train_s20_s

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s20_s/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s20_s \
    --select-top 20

python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_s \
    --model models/yolo11s.pt \
    --device mps \
    --output ./workspace/train_s50_s

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s50_s/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s50_s \
    --select-top 20



python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s20 \
    --model models/yolo11m.pt \
    --device mps \
    --output ./workspace/train_s20_m

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s20_m/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s20_m \
    --select-top 20

# 샘플 50개, pv string 및 pv panel 이외 지정, m 모델 사용
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m \
    --model models/yolo11m.pt \
    --device mps \
    --output ./workspace/train_s50_m

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s50_m/weights-2/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s50_m \
    --select-top 20

# 샘플 50개, pv string 및 pv panel만 지정, m 모델 사용
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m_2 \
    --model models/yolo11m.pt \
    --device cuda \
    --output ./workspace/train_s50_m_2

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s50_m_2/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s50_m_2 \
    --select-top 20

# 샘플 50개, pv string 및 pv panel, other 지정, m 모델 사용
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m_3 \
    --model models/yolo11m.pt \
    --epochs 100 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s50_m_3

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s50_m_3/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s50_m_3 \
    --select-top 20

# 샘플 50개, pv string 및 pv panel, other 지정, nagative 추가, m 모델 사용
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s50_m_4 \
    --model models/yolo11m.pt \
    --epochs 100 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s50_m_4

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s50_m_4/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s50_m_4 \
    --select-top 20

# 샘플 100개, pv string 및 pv panel, other 지정, nagative 추가, m 모델 사용
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s100_m \
    --model models/yolo11m.pt \
    --epochs 100 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s100_m

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s100_m/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s100_m \
    --select-top 20

# 샘플 100개, pv string 및 pv panel, other 지정, nagative 추가, m 모델 사용
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s100_m \
    --model models/yolo11m.pt \
    --epochs 200 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s100_m_2

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s100_m_2/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s100_m_2 \
    --select-top 20

# 샘플 200개, pv string 및 pv panel, other 지정, nagative 추가, m 모델 사용 (스트링 조건을 혼용)
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m \
    --model models/yolo11m.pt \
    --epochs 400 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_m/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_m \
    --select-top 20

# 샘플 200개, pv string 및 pv panel, other 지정, nagative 추가, m 모델 사용 (스트링 조건을 혼용, 배치 8)
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m \
    --model models/yolo11m.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m_b8

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_m_b8/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_m \
    --select-top 20

# 샘플 200개, pv string 및 pv panel, other 지정, nagative 추가, m 모델 사용 (스트링 조건을 명확히)
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m_2 \
    --model models/yolo11m.pt \
    --epochs 400 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m_2

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_m_2/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_m_2 \
    --select-top 20

# 샘플 200개, pv string 및 pv panel, other 지정, nagative 추가, m 모델 사용 (스트링 조건을 명확히, 배치 8)
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m_2 \
    --model models/yolo11m.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m_2_b8

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_m_2_b8/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_m_2 \
    --select-top 20

# 샘플 100개, pv string 및 pv panel, other, defect 지정, nagative 추가, m 모델 사용 (스트링 조건을 명확히, 배치 4)
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s100_m_d \
    --model models/yolo11m.pt \
    --epochs 200 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s100_m_d

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s100_m_d/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s100_m_d \
    --select-top 20

# 샘플 100개, pv string 및 pv panel, other, defect 지정, nagative 추가, m 모델 사용, MacOS에서 학습 (device=mps) <- 중간에 중지시킴
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s100_m_d \
    --model models/yolo11m.pt \
    --epochs 200 \
    --device mps \
    --output ./workspace/train_s100_m_d

# 샘플 200개, pv string 및 pv panel, other, defect 지정, nagative 추가, m 모델 사용 (스트링 조건을 명확히, 배치 8)
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_m_d \
    --model models/yolo11m.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_m_d

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_m_d/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_m_d \
    --select-top 20

# 샘플 200개, pv string 및 pv panel, other, defect 지정, nagative 추가, x 모델 사용 (스트링 조건을 명확히, 배치 4)
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_x_d \
    --model models/yolo11x.pt \
    --epochs 400 \
    --batch 4 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_x_d

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_x_d/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_x_d \
    --select-top 20

# confidence 2.0 -> 1.0으로 변경한 후 predict
python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_x_d/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_x_d \
    --conf 0.1 \
    --select-top 20

# 샘플 200개, pv string 및 pv panel, other, defect 지정, nagative 추가, x 모델 사용 (스트링 조건을 명확히, 배치 4)
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_x_2_d \
    --model models/yolo11x.pt \
    --epochs 400 \
    --batch 4 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_x_2_d

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_x_2_d/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_x_2_d \
    --select-top 20

# 샘플 200개, pv string 및 pv panel, other, defect 지정, nagative 추가, l 모델 사용 (스트링 조건을 명확히, 배치 8)
python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_l_d \
    --model models/yolo11l.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_l_d

python scripts/run_active_training.py seed \
    --images data/solar/images/RGB \
    --seed-labels ./workspace/labels_s200_l_2_d \
    --model models/yolo11l.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s200_l_2_d

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_l_2_d/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_l_2_d \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/images/RGB \
    --model ./runs/detect/workspace/train_s200_l_d/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s200_l_d \
    --select-top 20




python scripts/run_active_training.py iterate \
    --images data/solar/갈평저수지/RGB \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_galpyeong_l_d \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/에스엘에너지_사천시/RGB \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_sl_sacheon_l_d \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/옥산_1호/RGB \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_oksan_l_d \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/s300 \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s300 \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/s300_1 \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s300_1 \
    --select-top 20

# 
python scripts/run_training.py \
    --images data/solar/images/RGB \
    --labels-dir ./workspace/labels_s200_m_3_d \
    --visual-dir ./workspace/visualized \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/images/RGB \
    --labels-dir ./workspace/predict_s200_l_2_d/predicted_labels \
    --visual-dir ./workspace/visualized \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize



python scripts/run_training.py \
    --images data/solar/갈평저수지/RGB \
    --labels-dir ./workspace/predict_galpyeong_l_d/predicted_labels \
    --visual-dir ./workspace/visualized_galpyeong \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/에스엘에너지_사천시/RGB \
    --labels-dir ./workspace/predict_sl_sacheon_l_d/predicted_labels \
    --visual-dir ./workspace/visualized_sl_sacheon \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/옥산_1호/RGB \
    --labels-dir ./workspace/predict_oksan_l_d/predicted_labels \
    --visual-dir ./workspace/visualized_sl_oksan \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/s300 \
    --labels-dir ./workspace/labels_s300 \
    --visual-dir ./workspace/visualized_s300 \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize





python scripts/run_active_training.py seed \
    --images data/s300_l \
    --seed-labels ./workspace/labels_s300_l \
    --model models/yolo11l.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s300_l


python scripts/run_active_training.py iterate \
    --images data/solar/s300 \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s300_l \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/s300 \
    --model ./runs/detect/workspace/train_s300_l_2/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s300_l_2 \
    --select-top 20


python scripts/run_training.py \
    --images data/solar/s300 \
    --labels-dir ./workspace/labels_s300_2 \
    --visual-dir ./workspace/visualized_s300 \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/s300 \
    --labels-dir ./workspace/labels_s300_3 \
    --visual-dir ./workspace/visualized_s300 \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/s300_1 \
    --labels-dir ./workspace/labels_s300_3 \
    --visual-dir ./workspace/visualized_s300_1 \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize




python scripts/run_active_training.py iterate \
    --images data/solar/갈평저수지/RGB \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_galpyeong_l \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/에스엘에너지_사천시/RGB \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_sl_sacheon_l \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/옥산_1호/RGB \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_oksan_l \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/EWP-서오창IC-2/RGB \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_seochang_ic \
    --select-top 20

python scripts/run_training.py \
    --images data/solar/갈평저수지/RGB \
    --labels-dir ./workspace/predict_galpyeong_l/predicted_labels \
    --visual-dir ./workspace/visualized_galpyeong \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/에스엘에너지_사천시/RGB \
    --labels-dir ./workspace/predict_sl_sacheon_l/predicted_labels \
    --visual-dir ./workspace/visualized_sl_sacheon \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/옥산_1호/RGB \
    --labels-dir ./workspace/predict_oksan_l/predicted_labels \
    --visual-dir ./workspace/visualized_sl_oksan \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/EWP-서오창IC-2/RGB \
    --labels-dir ./workspace/predict_seochang_ic/predicted_labels \
    --visual-dir ./workspace/visualized_seochang_ic \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/환경관리/RGB \
    --labels-dir ./workspace/predict_environ/predicted_labels \
    --visual-dir ./workspace/visualized_environ \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_active_training.py iterate \
    --images data/solar/환경관리/RGB \
    --model ./runs/detect/workspace/train_s400_l/weights-3/weights/best.pt \
    --device mps \
    --output ./workspace/predict_environ \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/갈평저수지/RGB \
    --model ./runs/detect/workspace/train_s400_l/weights-3/weights/best.pt \
    --device mps \
    --output ./workspace/predict_galpyeong_l \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/에스엘에너지_사천시/RGB \
    --model ./runs/detect/workspace/train_s400_l/weights-3/weights/best.pt \
    --device mps \
    --output ./workspace/predict_sl_sacheon_l \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/옥산_1호/RGB \
    --model ./runs/detect/workspace/train_s400_l/weights-3/weights/best.pt \
    --device mps \
    --output ./workspace/predict_oksan_l \
    --select-top 20

python scripts/run_active_training.py iterate \
    --images data/solar/EWP-서오창IC-2/RGB \
    --model ./runs/detect/workspace/train_s400_l/weights-3/weights/best.pt \
    --device mps \
    --output ./workspace/predict_seochang_ic \
    --select-top 20


python scripts/run_active_training.py seed \
    --images data/s300_l \
    --seed-labels ./workspace/labels_s300_l \
    --model models/yolo11l.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s300_l

python scripts/run_active_training.py seed \
    --images data/s400_l \
    --seed-labels ./workspace/labels_s400_l \
    --model models/yolo11l.pt \
    --epochs 400 \
    --batch 8 \
    --device cuda \
    --amp True \
    --output ./workspace/train_s400_l

python scripts/run_active_training.py iterate \
    --images data/solar/s300 \
    --model ./runs/detect/workspace/train_s300_l/weights/weights/best.pt \
    --device mps \
    --output ./workspace/predict_s300_l \
    --select-top 20

python scripts/run_training.py \
    --images data/solar/s300 \
    --labels-dir ./workspace/labels_s300 \
    --visual-dir ./workspace/visualized_s300 \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize

python scripts/run_training.py \
    --images data/solar/s300 \
    --labels-dir ./workspace/predict_s300_l/predicted_labels \
    --visual-dir ./workspace/visualized_s300 \
    --strategy sam2 \
    --classes pv_string pv_module other anomaly \
    --steps visualize
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from ultralytics import YOLO

# 프로젝트를 editable 설치하지 않았을 때를 위해 src 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solar_thermal.labeling.active_learning import cmd_seed, cmd_iterate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # seed
    p = sub.add_parser("seed", help="수동 seed 라벨로 초기 YOLO 학습")
    p.add_argument("--images",      type=Path, required=True)
    p.add_argument("--seed-labels", type=Path, required=True)
    p.add_argument("--output",      type=Path, required=True)
    p.add_argument("--classes",     nargs="+", default=["pv_string", "pv_module", "other", "anomaly"])
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--imgsz",       type=int, default=1280)
    p.add_argument("--batch",       type=int, default=4)
    p.add_argument("--model",       default="models/yolo11n.pt")
    p.add_argument("--device",      default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--val-ratio",   type=float, default=0.2)
    p.add_argument("--amp",         type=bool, default=False)

    # predict
    p = sub.add_parser("predict", help="현재 모델로 예측 + uncertainty 점수")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--model",  type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conf",   type=float, default=0.2)
    p.add_argument("--imgsz",  type=int, default=1280)

    # select
    p = sub.add_parser("select", help="uncertainty 높은 샘플 human review 큐로")
    p.add_argument("--output",     type=Path, required=True,
                   help="predict 단계와 동일한 output 디렉토리")
    p.add_argument("--top-n",      type=int, default=20)

    # iterate
    p = sub.add_parser("iterate", help="predict + select 자동 실행")
    p.add_argument("--images",     type=Path, required=True)
    p.add_argument("--model",      type=Path, required=True)
    p.add_argument("--device",     default="mps", choices=["cpu", "cuda", "mps"])
    p.add_argument("--output",     type=Path, required=True)
    p.add_argument("--classes",    nargs="+", default=["pv_string", "pv_module", "other", "anomaly"])
    p.add_argument("--select-top", type=int, default=20)
    p.add_argument("--conf",       type=float, default=0.2)
    p.add_argument("--imgsz",      type=int, default=1280)

    args = ap.parse_args()

    if args.cmd == "seed":
        cmd_seed(
            args.images, args.seed_labels, args.output, args.classes,
            val_ratio=args.val_ratio, epochs=args.epochs,
            imgsz=args.imgsz, batch=args.batch, model=args.model,
            device=args.device,
            amp=args.amp,
        )
    elif args.cmd == "predict":
        cmd_predict(args.images, args.model, args.output,
                    conf=args.conf, imgsz=args.imgsz)
    elif args.cmd == "select":
        # Re-load samples from saved report
        report_path = args.output / "uncertainty_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                f"{report_path} 없음. 먼저 `predict` 실행하세요."
            )
        report = json.loads(report_path.read_text())
        # Reconstruct UncertaintySample for select
        samples = [
            UncertaintySample(
                image_path=args.output.parent / "unlabeled" / r["image"],
                predictions=[], score=r["score"],
            )
            for r in report
        ]
        cmd_select(samples, args.output, top_n=args.top_n)
    elif args.cmd == "iterate":
        cmd_iterate(
            args.images, args.model, args.output,
            classes=args.classes, select_top=args.select_top,
            conf=args.conf, imgsz=args.imgsz,
            device=args.device,
        )


if __name__ == "__main__":
    main()
