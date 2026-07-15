#include <Dynamixel2Arduino.h>

#define DXL_SERIAL   Serial3
#define DEBUG_SERIAL Serial

const int DXL_DIR_PIN = 84;

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

void testBaud(uint32_t baud)
{
  dxl.begin(baud);

  DEBUG_SERIAL.println();
  DEBUG_SERIAL.print("Testing baud: ");
  DEBUG_SERIAL.println(baud);

  bool found = false;

  for(int id = 0; id < 20; id++)
  {
    if(dxl.ping(id))
    {
      DEBUG_SERIAL.print("Found ID: ");
      DEBUG_SERIAL.println(id);

      found = true;
    }
  }

  if(!found)
  {
    DEBUG_SERIAL.println("No motor found");
  }
}

void setup()
{
  DEBUG_SERIAL.begin(115200);

  while(!DEBUG_SERIAL);

  dxl.setPortProtocolVersion(2.0);

  DEBUG_SERIAL.println("=== Dynamixel Scan Start ===");

  testBaud(57600);
  // testBaud(115200);
  // testBaud(1000000);
  // testBaud(2000000);

  DEBUG_SERIAL.println("=== Scan End ===");
}

void loop()
{
}