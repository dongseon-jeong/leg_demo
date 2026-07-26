#include <Dynamixel2Arduino.h>

#define DXL_SERIAL   Serial3
#define DEBUG_SERIAL Serial

const int DXL_DIR_PIN = 84;

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

using namespace ControlTableItem;

const float DXL_PROTOCOL_VERSION = 2.0;
const uint32_t BAUDRATE = 57600;

// 총 12개 모터 ID
const uint8_t IDS[28] = {
  1, 2, 3, 4, 5, 6,    // 왼발 (lbase, ll1, ll2, ll3, ll4, ll5)
  7, 8, 9, 10, 11, 12,  // 오른발 (rbase, rl1, rl2, rl3, rl4, rl5)
  13,14,15,16,17,18,
  19,20,21,22,23,24,
  25,26,27,28
};

// 관절 이름 매핑 (시리얼 모니터 가독성용)
const char* JOINT_NAMES[28] = {
  "L-Base", "L-Leg1", "L-Leg2", "L-Leg3", "L-Leg4", "L-Leg5",
  "R-Base", "R-Leg1", "R-Leg2", "R-Leg3", "R-Leg4", "R-Leg5",
  "neck-roll","neck-pitch",
  "lshd-pitch","lshd-yaw","lshd-roll",
  "lelbo","lwrist-pitch","lwrist-yaw","lgripper",
  "rshd-pitch","rshd-yaw","rshd-roll",
  "relbo","rwrist-pitch","rwrist-yaw","rgripper"
};

void setup()
{
  DEBUG_SERIAL.begin(115200);
  while(!DEBUG_SERIAL);

  DEBUG_SERIAL.println("========================================");
  DEBUG_SERIAL.println("=== DYNAMIXEL PRESENT POSITION CHECK ===");
  DEBUG_SERIAL.println("========================================");

  dxl.begin(BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  // 안전을 위해 토크를 완전히 끈 상태(IDLE)로 둡니다.
  // 손으로 다리를 움직이면서 각도 변화를 관찰할 수 있습니다.
  for(int i = 0; i < 12; i++)
  {
    uint8_t id = IDS[i];
    if(dxl.ping(id))
    {
      dxl.torqueOff(id); 
    }
    else
    {
      DEBUG_SERIAL.print("⚠️ Ping FAILED for ID: ");
      DEBUG_SERIAL.println(id);
    }
  }
  
  DEBUG_SERIAL.println("\n[INFO] All motors Torque-OFF. You can move joints by hand.");
  delay(1000);
}

void loop()
{
  DEBUG_SERIAL.println("\n--- Current Motor Positions (Raw Value) ---");

  for(int i = 0; i < 28; i++)
  {
    uint8_t id = IDS[i];
    
    // 모터의 현재 위치(Raw Value) 읽기
    int32_t present_position = dxl.getPresentPosition(id, UNIT_RAW);

    DEBUG_SERIAL.print("[ID: ");
    if(id < 10) DEBUG_SERIAL.print(" "); // 자릿수 맞춤 공백
    DEBUG_SERIAL.print(id);
    DEBUG_SERIAL.print("] ");
    
    // 관절 이름 출력
    DEBUG_SERIAL.print(JOINT_NAMES[i]);
    for(int s = strlen(JOINT_NAMES[i]); s < 8; s++) DEBUG_SERIAL.print(" "); // 정렬용 공백
    
    DEBUG_SERIAL.print(" -> Raw: ");
    DEBUG_SERIAL.println(present_position);
  }

  DEBUG_SERIAL.println("-------------------------------------------");
  
  // 1초마다 갱신하며 출력
  delay(1000); 
}