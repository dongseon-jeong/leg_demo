#include <Dynamixel2Arduino.h>
#include <IMU.h>

#define DXL_SERIAL   Serial3
#define DEBUG_SERIAL Serial
#define RPI_SERIAL   Serial   // 라즈베리파이 USB 시리얼 연결 (OpenCR은 Serial이 USB 가상 시리얼입니다)

cIMU imu;
const int DXL_DIR_PIN = 84;
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

using namespace ControlTableItem;

const float DXL_PROTOCOL_VERSION = 2.0;
const uint32_t BAUDRATE = 57600;

const uint8_t IDS[28] = {
  1, 2, 3, 4, 5, 6,    // 왼발 (1~6)
  7, 8, 9, 10, 11, 12,  // 오른발 (7~12)
  13,14,
  15,16,17,18,19,20,21,
  22,23,24,25,26,27,28
};

int32_t DYNAMIC_CENTER_POS[28] = {0, };


const float URDF_MIN_RAD[28] = {
  -1.57,   -1.57,   -1.57,    0.0,   -0.43, -1.04, // 왼발 (1번: -1.57)
  0.0,     -0.43,   -0.43,   -1.39,  -1.57, -1.57,  // 오른발 (7번: 0.0)
  -1.570796,-1.570796,
  -3.141592,-1.570796,-1.570796,
  -2.094395,-1.570796,-0.349066,-5.2360,
  -3.141592,-0.349066,-1.570796,
  -2.094395,-1.570796,-0.349066,-5.2360

};

const float URDF_MAX_RAD[28] = {
  0.0,    0.43,     0.43,     1.39,   1.57,  1.57,  // 왼발 (1번: 0.0)
  1.57,   1.57,     1.57,     0.0,    0.43,  1.04,   // 오른발 (7번: 1.57)
  1.570796,1.570796,
  3.141592,0.349066,1.570796,
  2.094395,1.570796,0.349066,0.8727,
  3.141592,1.570796,1.570796,
  2.094395,1.570796,0.349066,0.8727

};





const float RAD_TO_DXL_STEP = 651.74;

// void setup()
// {
//   DEBUG_SERIAL.begin(115200);
//   while(!DEBUG_SERIAL);

//   DEBUG_SERIAL.println("==============================================");
//   DEBUG_SERIAL.println("===  PERFECT DIRECTION SYMMETRIC CONTROLLER ===");
//   DEBUG_SERIAL.println("==============================================");

//   dxl.begin(BAUDRATE);
//   dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

//   DEBUG_SERIAL.println(">> Capturing current positions...");
//   for(int i = 0; i < 12; i++)
//   {
//     uint8_t id = IDS[i];
//     if(dxl.ping(id)) {
//       DYNAMIC_CENTER_POS[i] = dxl.getPresentPosition(id, UNIT_RAW);
//       DEBUG_SERIAL.print(" Motor ID ["); DEBUG_SERIAL.print(id);
//       DEBUG_SERIAL.print("] Center (0 rad) -> Raw: "); DEBUG_SERIAL.println(DYNAMIC_CENTER_POS[i]);
//     } else {
//       DEBUG_SERIAL.print("⚠️ Ping FAILED for ID: "); DEBUG_SERIAL.println(id);
//       DYNAMIC_CENTER_POS[i] = 2048; 
//     }
//   }

//   DEBUG_SERIAL.println("\n>> Enabling Extended Position Mode...");
//   for(int i = 0; i < 12; i++)
//   {
//     uint8_t id = IDS[i];
//     dxl.torqueOff(id);
//     dxl.setOperatingMode(id, OP_EXTENDED_POSITION); 

//     dxl.writeControlTableItem(PROFILE_VELOCITY, id, 30);
//     dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 10);
//     dxl.writeControlTableItem(POSITION_P_GAIN, id, 850); 
//     dxl.writeControlTableItem(POSITION_D_GAIN, id, 0); 

//     dxl.torqueOn(id);
//     dxl.setGoalPosition(id, (uint32_t)DYNAMIC_CENTER_POS[i], UNIT_RAW);
//     delay(20);
//   }
//   DEBUG_SERIAL.println(">> System Ready.");
// }

void setup()
{
  DEBUG_SERIAL.begin(115200); // 파이썬 통신 속도와 일치시킵니다.
  
  dxl.begin(BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  for(int i = 0; i < 28; i++) {
    uint8_t id = IDS[i];
    if(dxl.ping(id)) {
      DYNAMIC_CENTER_POS[i] = dxl.getPresentPosition(id, UNIT_RAW);
    } else {
      DYNAMIC_CENTER_POS[i] = 2048; 
    }
  }

  for(int i = 0; i < 28; i++) {
    uint8_t id = IDS[i];
    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_EXTENDED_POSITION); 
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 30);
    dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 10);
    dxl.writeControlTableItem(POSITION_P_GAIN, id, 850); 
    dxl.writeControlTableItem(POSITION_D_GAIN, id, 0); 
    dxl.torqueOn(id);
    dxl.setGoalPosition(id, (uint32_t)DYNAMIC_CENTER_POS[i], UNIT_RAW);
    delay(20);
  }
  imu.begin();
  delay(100);
}

// void loop()
// {
//   if (DEBUG_SERIAL.available() > 0)
//   {
//     String inputString = DEBUG_SERIAL.readStringUntil('\n');
//     inputString.trim();

//     if (inputString.length() == 0) return;

//     if (inputString == "0") {
//       DEBUG_SERIAL.println("\n🚨 [RESET] Returning ALL to Center.");
//       for(int i = 0; i < 12; i++) {
//         dxl.setGoalPosition(IDS[i], (uint32_t)DYNAMIC_CENTER_POS[i], UNIT_RAW);
//       }
//       return;
//     }

//     DEBUG_SERIAL.println("\n--- Processing Symmetrical Command ---");

//     int currentPos = 0;
//     int nextSlash = 0;

//     while (currentPos < inputString.length()) {
//       nextSlash = inputString.indexOf('/', currentPos);
      
//       String token;
//       if (nextSlash == -1) {
//         token = inputString.substring(currentPos);
//         currentPos = inputString.length();
//       } else {
//         token = inputString.substring(currentPos, nextSlash);
//         currentPos = nextSlash + 1;
//       }

//       token.trim();
//       if (token.length() == 0) continue;

//       int commaIndex = token.indexOf(',');
//       if (commaIndex == -1) {
//         DEBUG_SERIAL.print("⚠️ Skip Invalid Token: "); DEBUG_SERIAL.println(token);
//         continue;
//       }

//       String idPart = token.substring(0, commaIndex);
//       String offsetPart = token.substring(commaIndex + 1);

//       int targetID = idPart.toInt();
//       int32_t offsetInput = offsetPart.toInt();

//       if (targetID < 1 || targetID > 12) {
//         DEBUG_SERIAL.print("⚠️ Skip Invalid ID: "); DEBUG_SERIAL.println(targetID);
//         continue;
//       }

//       int idx = targetID - 1;
//       int32_t center = DYNAMIC_CENTER_POS[idx];

//       // 1. 기본 URDF 라디안 기반 제한값 계산
//       int32_t min_offset_limit = (int32_t)(URDF_MIN_RAD[idx] * RAD_TO_DXL_STEP);
//       int32_t max_offset_limit = (int32_t)(URDF_MAX_RAD[idx] * RAD_TO_DXL_STEP);


//       // if (targetID == 11|| targetID == 12) {
//       //   offsetInput = -offsetInput; // 입력 부호 반전 (원하는 방향 전환)
        
//       //   // ★ 9번은 이미 배열이 양수(0.0 ~ 1.57)로 완벽하므로 가드 장벽을 반전시키지 않고 그대로 유지합니다!
//       //   DEBUG_SERIAL.print(" [INFO] ID 9 Direction Inverted -> Offset: ");
//       //   DEBUG_SERIAL.println(offsetInput);
//       // }

//       // 3. URDF 오프셋 범위 내로 가드 클램핑 (이제 9번은 0 ~ 1023 가드를 정상 적용받습니다)
//       if (offsetInput < min_offset_limit) {
//         DEBUG_SERIAL.print(" [URDF LIMIT] ID "); DEBUG_SERIAL.print(targetID);
//         DEBUG_SERIAL.print(" Min Limit! Bound to: "); DEBUG_SERIAL.println(min_offset_limit);
//         offsetInput = min_offset_limit;
//       }
//       if (offsetInput > max_offset_limit) {
//         DEBUG_SERIAL.print(" [URDF LIMIT] ID "); DEBUG_SERIAL.print(targetID);
//         DEBUG_SERIAL.print(" Max Limit! Bound to: "); DEBUG_SERIAL.println(max_offset_limit);
//         offsetInput = max_offset_limit;
//       }

//       // 4. 최종 타겟 계산 및 Extended Position 모드에 맞게 전송
//       int32_t finalTarget = center + offsetInput;

//       DEBUG_SERIAL.print(" ▶ [EXECUTE] ID: "); DEBUG_SERIAL.print(targetID);
//       DEBUG_SERIAL.print(" -> Offset: "); DEBUG_SERIAL.print(offsetInput);
//       DEBUG_SERIAL.print(" -> Goal Raw: "); DEBUG_SERIAL.println(finalTarget);

//       // Extended Mode에서는 부호 있는 32비트 값(finalTarget)을 그대로 던져주면 모터가 알아서 경계선을 넘나듭니다.
//       dxl.setGoalPosition(targetID, finalTarget, UNIT_RAW);
//     }
//     DEBUG_SERIAL.println("--------------------------------");
//   }
// }



void loop()
{
  // 라즈베리파이(RPI_SERIAL)로부터 데이터 수신
  if (RPI_SERIAL.available() > 0)
  {
    String inputString = RPI_SERIAL.readStringUntil('\n');
    inputString.trim();

    if (inputString.length() == 0) return;

    if (inputString == "0") {
      for(int i = 0; i < 28; i++) {
        dxl.setGoalPosition(IDS[i], (uint32_t)DYNAMIC_CENTER_POS[i], UNIT_RAW);
      }
      RPI_SERIAL.println("ACK:RESET");
      return;
    }

    int currentPos = 0;
    int nextSlash = 0;

    while (currentPos < inputString.length()) {
      nextSlash = inputString.indexOf('/', currentPos);
      
      String token;
      if (nextSlash == -1) {
        token = inputString.substring(currentPos);
        currentPos = inputString.length();
      } else {
        token = inputString.substring(currentPos, nextSlash);
        currentPos = nextSlash + 1;
      }

      token.trim();
      if (token.length() == 0) continue;

      int commaIndex = token.indexOf(',');
      if (commaIndex == -1) continue;

      String idPart = token.substring(0, commaIndex);
      String offsetPart = token.substring(commaIndex + 1);

      int targetID = idPart.toInt();
      int32_t offsetInput = offsetPart.toInt();

      if (targetID < 1 || targetID > 28) continue;

      int idx = targetID - 1;
      int32_t center = DYNAMIC_CENTER_POS[idx];

      int32_t min_offset_limit = (int32_t)(URDF_MIN_RAD[idx] * RAD_TO_DXL_STEP);
      int32_t max_offset_limit = (int32_t)(URDF_MAX_RAD[idx] * RAD_TO_DXL_STEP);

      // 물리적 방향/한계 분기문
      if (targetID == 3|| targetID == 4|| targetID == 5|| targetID == 6||targetID == 9|| targetID == 10|| targetID == 11|| targetID == 12) {
        offsetInput = -offsetInput; 
      }


      // 가드 클램핑
      if (offsetInput < min_offset_limit) offsetInput = min_offset_limit;
      if (offsetInput > max_offset_limit) offsetInput = max_offset_limit;

      int32_t finalTarget = center + offsetInput;
      dxl.setGoalPosition(targetID, finalTarget, UNIT_RAW);
    }
    // 라즈베리파이에게 명령을 잘 처리했다고 응답 전송 (파이썬 블로킹 방지용)
    RPI_SERIAL.println("ACK:OK");
  }

  imu.update();
  // 2. IMU 데이터 읽기 및 라즈베리파이로 송출 (추가할 부분)
  // 50Hz~100Hz 주기로 보낼 수 있도록 타이머 처리
  // float roll  = imu.rpy[0];  // Index 0 : Roll
  // float pitch = imu.rpy[1];  // Index 1 : Pitch
  // float yaw   = imu.rpy[2];  // Index 2 : Yaw
  // // "IMU:롤값,피치값,요값" 형태로 한 줄씩 출력 (\n)
  // Serial.print("IMU:");
  // Serial.print(roll); Serial.print(",");
  // Serial.print(pitch); Serial.print(",");
  // Serial.println(yaw);

  float raw_roll  = imu.rpy[0];
  float raw_pitch = imu.rpy[1];
  float raw_yaw   = imu.rpy[2];

  // 시계방향 90도 회전 보정
  float corrected_roll  = -raw_roll;
  float corrected_pitch = -raw_pitch;
  float corrected_yaw   =  raw_yaw;

  // yaw 범위 정규화 (-180 ~ 180)
  if (corrected_yaw < -180.0f) corrected_yaw += 360.0f;
  if (corrected_yaw >  180.0f) corrected_yaw -= 360.0f;

  Serial.print("IMU:");
  Serial.print(corrected_roll);  Serial.print(",");
  Serial.print(corrected_pitch); Serial.print(",");
  Serial.println(corrected_yaw);

  delay(10);
  
}