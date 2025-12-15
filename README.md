

# 기획

- 로봇
	- op3 leg + 7dof arm
- 최종로봇목표
	- 하체는 밸런싱 보드에서 균형 잡고 상체는 책상 위 주사위를 들어 번호 순으로 여러 개를 쌓는 작업을 구현 목표
- 학습
	- 하체는 isaaclab 강화학습
	- 상체는 lerobot 모방학습
- 추론
	- ros로 두 모델의 출력 밸런스 관리
- 부품
	- 하체 다이나믹셀 XM430-W350-R x 12
	- 상체 다이나믹셀 XM430-W350-R x 7
	- 상체 usb 카메라(or depth)
	- opencr, 라즈베리파이
	- 리튬베터리




# 3d 프린팅 부품 설계

- onshape이나 fusion 사용
- op3의 stl 파일이 있으나 본래 부분은 금속소재로 플라스틱 출력 시의 강도가 예상되지 않아, 새로 적당한 두께로 설계
- 모터 stl를 불러와서 사용하고 다른 부분의 스케치가 없어서 대략적으로 스케치하여 모델링 

![이미지](./image/Pasted%20image%2020251129122659.png)




# 3d 조립 완성 및 단순구조 변환

![이미지](./image/Pasted%20image%2020251129122817.png)

- 단순 구조로 변환
	- 모터의 회전 조인트 및 링크 이름 설정 및 조인트 한계각도 설정
	- fusion2urdf repo에 유의사항을 체크하여 아래와 같이 변환해야 urdf 변환 오류가 발생안됨

![이미지](./image/Pasted%20image%2020251129123000.png)

https://github.com/dheena2k2/fusion2urdf-ros2
https://github.com/runtimerobotics/fusion360-urdf-ros2




# urdf 변환 및 usd 파일 생성

![이미지](./image/Pasted%20image%2020251129135616.png)

isaacsim에서 urdf import하여 inertial collision friction damping 등 설정하여 usd 저장

![이미지](./image/Pasted%20image%2020251129123345.png)




# isaaclab 걷기 학습

환경 생성
```bash
./isaaclab.sh --new
Task type : Externel
Project path : d:/making/dynamixel/leg
Project name : leg
Direct | single-agent
rsl_rel, skrl
AMP, PPO
```

- 기본으로 생성하면 다음과 같이 생성됨
- cart pole로 되어 있어 커스텀 코드로 수정

![이미지](./image/Pasted%20image%2020251129135716.png)

- usd 파일
leg\source\leg\leg\tasks\direct\leg\legs_cfg.py
```python
 
LEGS_CFG = ArticulationCfg(
    # ⬇ 최종 스테이지에서 로봇의 prim 경로 패턴
    #    /World/envs/env_0/world/legs, /World/envs/env_1/world/legs, ...
    prim_path="{ENV_REGEX_NS}/legs",
    spawn=sim_utils.UsdFileCfg(
        # 여긴 그대로 네 USD 파일 경로
        usd_path=f"D:/making/dynamixel/leg.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "legs_dxl": ImplicitActuatorCfg(
            joint_names_expr=[
                "lbase_joint",
                "rbase_joint",
                "ll1_joint",
                "rl1_joint",
                "ll2_joint",
                "rl2_joint",
                "ll3_joint",
                "rl3_joint",
                "ll4_joint",
                "rl4_joint",
                "ll5_joint",
                "rl5_joint",
            ],
            stiffness={".*": 25.0},
            damping={".*": 1.0},
            velocity_limit_sim={".*": 5.0},
        ),
    },
)

```

- 기본 환경 설정
leg\source\leg\leg\tasks\direct\leg\leg_env_cfg.py

- 액션 스페이스 및 보상 등 중요한 환경 요소 수정
leg\source\leg\leg\tasks\direct\leg\leg_env.py

- 모델 관련
leg\source\leg\leg\tasks\direct\leg\agents\rsl_rl_ppo_cfg.py
leg\source\leg\leg\tasks\direct\leg\agents\skrl_amp_cfg.yaml
leg\source\leg\leg\tasks\direct\leg\agents\skrl_ppo_cfg.yaml

- 그 외 확인할 것
leg\source\leg\leg\tasks\direct\leg\__init__.py
leg\scripts\rsl_rl\train.py

- 학습 실행
```bash
python scripts/rsl_rl/train.py --task=Template-Leg-Direct-v0 --num_envs=10000 --max_iterations=100000
```


![이미지](./image/Pasted%20image%2020251129123624.png)

```d
################################################################################
                       Learning iteration 149/150

                       Computation: 9299 steps/s (collection: 1.526s, learning 0.194s)
             Mean action noise std: 0.90
          Mean value_function loss: 9.9457
               Mean surrogate loss: -0.0034
                 Mean entropy loss: 15.6270
                       Mean reward: -11.38
               Mean episode length: 1.00
--------------------------------------------------------------------------------
                   Total timesteps: 2400000
                    Iteration time: 1.72s
                      Time elapsed: 00:04:22
                               ETA: 00:00:01

[307.600s] Simulation App Shutting Down
```

- 추론
```bash
python scripts/rsl_rl/play.py --task Template-Leg-Direct-v0 --num_envs 1 --checkpoint logs/rsl_rl/leg/2025-12-15-22-00-00/model_39400.pt
```


# simul ros 통신 구현




# 3d 프린팅 조립




# real ros 통신 구현




# 참고
https://github.com/ROBOTIS-GIT/ROBOTIS-OP3-Common
https://www.robotis.com/service/downloadpage.php?ca_id=70
https://emanual.robotis.com/docs/en/platform/op3/robotis_ros_packages/#robotis-ros-packages
https://www.youtube.com/watch?v=tQziqSx-F80&t=1970s