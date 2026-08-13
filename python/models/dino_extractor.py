import onnxruntime as ort
import numpy as np
import cv2
import os

class DinoExtractor:
    def __init__(self, model_path="model/model.onnx"):
        # 1. 제트슨 보드 최적화 가속 설정 (TensorRT 에러 우회 및 CUDA 우선)
        # TensorrtExecutionProvider는 특정 연산자(SplitToSequence) 호환성 문제로 제외
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
        
        # 모델 파일 존재 여부 확인
        if not os.path.exists(model_path):
            # Fallback for local testing if needed, but prioritize exact path
            if os.path.exists("model/model_sim.onnx"):
                model_path = "model/model_sim.onnx"
            else:
                 raise FileNotFoundError(f"모델을 찾을 수 없습니다: {os.path.abspath(model_path)}")
                 
        try:
            print(f"[{model_path}] 로딩 중... CUDA 가속 엔진을 초기화합니다.")
            # 2. ONNX Runtime 세션 생성
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # 실제 활성화된 엔진 확인
            active_providers = self.session.get_providers()
            print(f"✅ 활성화된 엔진: {active_providers[0]}")
        except Exception as e:
            print(f"❌ 엔진 초기화 실패: {e}")
            print("CPU 모드로 전환하여 시도합니다.")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # SAT-493M 전용 정규화 파라미터
        self.mean = np.array([0.430, 0.411, 0.296]).reshape(1, 3, 1, 1).astype(np.float32)
        self.std = np.array([0.213, 0.156, 0.143]).reshape(1, 3, 1, 1).astype(np.float32)

    def extract(self, img_path, size=256):
            """이미지에서 DINO 특징 벡터 추출 및 L2 정규화"""
            from PIL import Image
            
            # 1. 이미지 로드 및 전처리
            img = Image.open(img_path).convert('RGB')
            img_res = img.resize((size, size), Image.BICUBIC) 
            
            img_array = np.array(img_res, dtype=np.float32) / 255.0
            
            # SAT-493M 정규화 파라미터 적용
            mean = np.array([0.430, 0.411, 0.296]).reshape(1, 1, 3)
            std = np.array([0.213, 0.156, 0.143]).reshape(1, 1, 3)
            img_norm = (img_array - mean) / std
            
            img_tensor = img_norm.transpose(2, 0, 1)[None].astype(np.float32)
            
            # 2. 추론
            inputs = {self.session.get_inputs()[0].name: img_tensor}
            outputs = self.session.run(None, inputs)
            
            # [수정 포인트] 261개 토큰 중 0번(CLS) 토큰만 선택!
            # shape: (1, 261, 1024) -> (1024,)
            feat = outputs[0][0, 0, :] 
            
            # 3. L2 정규화 (코사인 유사도 연산을 위해 필수)
            norm = np.linalg.norm(feat)
            return feat / (norm + 1e-6)