/*===============================================================*/
/*                                                               */
/*                      face_detect.cpp                          */
/*                                                               */
/*     Hardware function for the Face Detection application.     */
/*                                                               */
/*===============================================================*/

#include "../host/typedefs.h"

#include "haar_mapping.h"

uint5_t get_bank(uint10_t addr)
{
  #pragma HLS inline
  return bank_mapping[addr];
}

uint5_t get_offset(uint10_t addr)
{
  #pragma HLS inline
  return offset_mapping[addr];
}

uint18_t get_data0(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;
  switch (offset)
  {
    case 0: a = aa[19 ]; break;
    case 1: a = aa[29 ]; break;
    case 2: a = aa[52 ]; break;
    case 3: a = aa[100]; break;
    case 4: a = aa[132]; break;
    case 5: a = aa[161]; break;
    case 6: a = aa[193]; break;
    case 7: a = aa[220]; break;
    case 8: a = aa[239]; break;
    case 9: a = aa[253]; break;
    case 10:a = aa[284]; break;
    case 11:a = aa[309]; break;
    case 12:a = aa[362]; break;
    case 13:a = aa[385]; break;
    case 14:a = aa[396]; break;
    case 15:a = aa[447]; break;
    case 16:a = aa[448]; break;
    case 17:a = aa[449]; break;
    case 18:a = aa[451]; break;
    case 19:a = aa[466]; break;
    case 20:a = aa[492]; break;
    case 21:a = aa[531]; break;
    case 22:a = aa[562]; break;
    default: a = 0; break;
  }; 

  return a;
}

uint18_t get_data1(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;
  switch (offset)
  {
    case 0: a = aa[7  ]; break;
    case 1: a = aa[18 ]; break;
    case 2: a = aa[65 ]; break;
    case 3: a = aa[72 ]; break;
    case 4: a = aa[148]; break;
    case 5: a = aa[149]; break;
    case 6: a = aa[153]; break;
    case 7: a = aa[164]; break;
    case 8: a = aa[191]; break;
    case 9: a = aa[208]; break;
    case 10:a = aa[229]; break;
    case 11:a = aa[240]; break;
    case 12:a = aa[251]; break;
    case 13:a = aa[256]; break;
    case 14:a = aa[280]; break;
    case 15:a = aa[384]; break;
    case 16:a = aa[450]; break;
    case 17:a = aa[478]; break;
    case 18:a = aa[506]; break;
    case 19:a = aa[517]; break;
    case 20:a = aa[527]; break;
    case 21:a = aa[542]; break;
    case 22:a = aa[554]; break;
    case 23:a = aa[573]; break;
    case 24:a = aa[576]; break;
    case 25:a = aa[621]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data2(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;
  switch (offset)
  {
    case 0: a = aa[9  ]; break;
    case 1: a = aa[20 ]; break;
    case 2: a = aa[28 ]; break;
    case 3: a = aa[48 ]; break;
    case 4: a = aa[74 ]; break;
    case 5: a = aa[97 ]; break;
    case 6: a = aa[168]; break;
    case 7: a = aa[187]; break;
    case 8: a = aa[188]; break;
    case 9: a = aa[213]; break;
    case 10:a = aa[233]; break;
    case 11:a = aa[260]; break;
    case 12:a = aa[261]; break;
    case 13:a = aa[277]; break;
    case 14:a = aa[303]; break;
    case 15:a = aa[314]; break;
    case 16:a = aa[329]; break;
    case 17:a = aa[356]; break;
    case 18:a = aa[375]; break;
    case 19:a = aa[376]; break;
    case 20:a = aa[452]; break;
    case 21:a = aa[499]; break;
    case 22:a = aa[519]; break;
    case 23:a = aa[529]; break;
    case 24:a = aa[536]; break;
    case 25:a = aa[551]; break;
    case 26:a = aa[567]; break;
    case 27:a = aa[597]; break;
    case 28:a = aa[615]; break;
    default: a = 0; break;
  };
  return a;
}

uint18_t get_data3(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;
  switch (offset)
  {
    case 0: a = aa[41 ]; break;
    case 1: a = aa[56 ]; break;
    case 2: a = aa[79 ]; break;
    case 3: a = aa[96 ]; break;
    case 4: a = aa[109]; break;
    case 5: a = aa[141]; break;
    case 6: a = aa[155]; break;
    case 7: a = aa[201]; break;
    case 8: a = aa[249]; break;
    case 9: a = aa[263]; break;
    case 10:a = aa[293]; break;
    case 11:a = aa[322]; break;
    case 12:a = aa[383]; break;
    case 13:a = aa[394]; break;
    case 14:a = aa[408]; break;
    case 15:a = aa[415]; break;
    case 16:a = aa[428]; break;
    case 17:a = aa[445]; break;
    case 18:a = aa[459]; break;
    case 19:a = aa[479]; break;
    case 20:a = aa[532]; break;
    case 21:a = aa[564]; break;
    case 22:a = aa[575]; break;
    case 23:a = aa[598]; break;
    case 24:a = aa[611]; break;
    default: a = 0; break;
  };
  return a;
}

uint18_t get_data4(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[24 ]; break;
    case 1: a = aa[34 ]; break;
    case 2: a = aa[47 ]; break;
    case 3: a = aa[58 ]; break;
    case 4: a = aa[105]; break;
    case 5: a = aa[128]; break;
    case 6: a = aa[162]; break;
    case 7: a = aa[179]; break;
    case 8: a = aa[218]; break;
    case 9: a = aa[226]; break;
    case 10:a = aa[346]; break;
    case 11:a = aa[364]; break;
    case 12:a = aa[369]; break;
    case 13:a = aa[388]; break;
    case 14:a = aa[406]; break;
    case 15:a = aa[425]; break;
    case 16:a = aa[440]; break;
    case 17:a = aa[453]; break;
    case 18:a = aa[458]; break;
    case 19:a = aa[486]; break;
    case 20:a = aa[510]; break;
    case 21:a = aa[552]; break;
    case 22:a = aa[594]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data5(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[23 ]; break;
    case 1: a = aa[53 ]; break;
    case 2: a = aa[94 ]; break;
    case 3: a = aa[95 ]; break;
    case 4: a = aa[101]; break;
    case 5: a = aa[139]; break;
    case 6: a = aa[171]; break;
    case 7: a = aa[180]; break;
    case 8: a = aa[222]; break;
    case 9: a = aa[267]; break;
    case 10:a = aa[275]; break;
    case 11:a = aa[311]; break;
    case 12:a = aa[312]; break;
    case 13:a = aa[333]; break;
    case 14:a = aa[365]; break;
    case 15:a = aa[390]; break;
    case 16:a = aa[397]; break;
    case 17:a = aa[409]; break;
    case 18:a = aa[410]; break;
    case 19:a = aa[426]; break;
    case 20:a = aa[443]; break;
    case 21:a = aa[463]; break;
    case 22:a = aa[537]; break;
    case 23:a = aa[571]; break;
    case 24:a = aa[599]; break;
    case 25:a = aa[608]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data6(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;
  switch (offset)
  {
    case 0: a = aa[15 ]; break;
    case 1: a = aa[42 ]; break;
    case 2: a = aa[55 ]; break;
    case 3: a = aa[122]; break;
    case 4: a = aa[138]; break;
    case 5: a = aa[177]; break;
    case 6: a = aa[204]; break;
    case 7: a = aa[215]; break;
    case 8: a = aa[228]; break;
    case 9: a = aa[231]; break;
    case 10:a = aa[250]; break;
    case 11:a = aa[287]; break;
    case 12:a = aa[307]; break;
    case 13:a = aa[308]; break;
    case 14:a = aa[366]; break;
    case 15:a = aa[391]; break;
    case 16:a = aa[411]; break;
    case 17:a = aa[424]; break;
    case 18:a = aa[435]; break;
    case 19:a = aa[468]; break;
    case 20:a = aa[497]; break;
    case 21:a = aa[539]; break;
    case 22:a = aa[555]; break;
    case 23:a = aa[609]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data7(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[38 ]; break;
    case 1: a = aa[82 ]; break;
    case 2: a = aa[93 ]; break;
    case 3: a = aa[135]; break;
    case 4: a = aa[159]; break;
    case 5: a = aa[195]; break;
    case 6: a = aa[212]; break;
    case 7: a = aa[237]; break;
    case 8: a = aa[238]; break;
    case 9: a = aa[258]; break;
    case 10:a = aa[269]; break;
    case 11:a = aa[283]; break;
    case 12:a = aa[310]; break;
    case 13:a = aa[328]; break;
    case 14:a = aa[355]; break;
    case 15:a = aa[421]; break;
    case 16:a = aa[427]; break;
    case 17:a = aa[465]; break;
    case 18:a = aa[523]; break;
    case 19:a = aa[547]; break;
    case 20:a = aa[557]; break;
    case 21:a = aa[570]; break;
    case 22:a = aa[593]; break;
    case 23:a = aa[606]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data8(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0:a = aa[0]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data9(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[14 ]; break;
    case 1: a = aa[46 ]; break;
    case 2: a = aa[119]; break;
    case 3: a = aa[147]; break;
    case 4: a = aa[150]; break;
    case 5: a = aa[151]; break;
    case 6: a = aa[163]; break;
    case 7: a = aa[185]; break;
    case 8: a = aa[198]; break;
    case 9: a = aa[242]; break;
    case 10:a = aa[262]; break;
    case 11:a = aa[286]; break;
    case 12:a = aa[302]; break;
    case 13:a = aa[315]; break;
    case 14:a = aa[340]; break;
    case 15:a = aa[343]; break;
    case 16:a = aa[358]; break;
    case 17:a = aa[359]; break;
    case 18:a = aa[429]; break;
    case 19:a = aa[481]; break;
    case 20:a = aa[489]; break;
    case 21:a = aa[507]; break;
    case 22:a = aa[520]; break;
    case 23:a = aa[525]; break;
    case 24:a = aa[572]; break;
    case 25:a = aa[577]; break;
    case 26:a = aa[591]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data10(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[6  ]; break;
    case 1: a = aa[40 ]; break;
    case 2: a = aa[103]; break;
    case 3: a = aa[146]; break;
    case 4: a = aa[173]; break;
    case 5: a = aa[174]; break;
    case 6: a = aa[232]; break;
    case 7: a = aa[268]; break;
    case 8: a = aa[279]; break;
    case 9: a = aa[341]; break;
    case 10:a = aa[374]; break;
    case 11:a = aa[386]; break;
    case 12:a = aa[405]; break;
    case 13:a = aa[420]; break;
    case 14:a = aa[439]; break;
    case 15:a = aa[471]; break;
    case 16:a = aa[488]; break;
    case 17:a = aa[509]; break;
    case 18:a = aa[526]; break;
    case 19:a = aa[612]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data11(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[22 ]; break;
    case 1: a = aa[45 ]; break;
    case 2: a = aa[102]; break;
    case 3: a = aa[136]; break;
    case 4: a = aa[137]; break;
    case 5: a = aa[156]; break;
    case 6: a = aa[157]; break;
    case 7: a = aa[183]; break;
    case 8: a = aa[184]; break;
    case 9: a = aa[210]; break;
    case 10:a = aa[221]; break;
    case 11:a = aa[235]; break;
    case 12:a = aa[291]; break;
    case 13:a = aa[324]; break;
    case 14:a = aa[344]; break;
    case 15:a = aa[353]; break;
    case 16:a = aa[377]; break;
    case 17:a = aa[398]; break;
    case 18:a = aa[417]; break;
    case 19:a = aa[418]; break;
    case 20:a = aa[454]; break;
    case 21:a = aa[511]; break;
    case 22:a = aa[524]; break;
    case 23:a = aa[540]; break;
    case 24:a = aa[559]; break;
    case 25:a = aa[584]; break;
    case 26:a = aa[613]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data12(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[5  ]; break;
    case 1: a = aa[39 ]; break;
    case 2: a = aa[54 ]; break;
    case 3: a = aa[61 ]; break;
    case 4: a = aa[75 ]; break;
    case 5: a = aa[76 ]; break;
    case 6: a = aa[106]; break;
    case 7: a = aa[140]; break;
    case 8: a = aa[165]; break;
    case 9: a = aa[209]; break;
    case 10:a = aa[245]; break;
    case 11:a = aa[246]; break;
    case 12:a = aa[316]; break;
    case 13:a = aa[347]; break;
    case 14:a = aa[412]; break;
    case 15:a = aa[413]; break;
    case 16:a = aa[444]; break;
    case 17:a = aa[464]; break;
    case 18:a = aa[490]; break;
    case 19:a = aa[530]; break;
    case 20:a = aa[534]; break;
    case 21:a = aa[535]; break;
    case 22:a = aa[560]; break;
    case 23:a = aa[586]; break;
    case 24:a = aa[618]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data13(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[1  ]; break;
    case 1: a = aa[27 ]; break;
    case 2: a = aa[59 ]; break;
    case 3: a = aa[87 ]; break;
    case 4: a = aa[118]; break;
    case 5: a = aa[131]; break;
    case 6: a = aa[167]; break;
    case 7: a = aa[175]; break;
    case 8: a = aa[247]; break;
    case 9: a = aa[319]; break;
    case 10:a = aa[334]; break;
    case 11:a = aa[335]; break;
    case 12:a = aa[371]; break;
    case 13:a = aa[387]; break;
    case 14:a = aa[395]; break;
    case 15:a = aa[414]; break;
    case 16:a = aa[442]; break;
    case 17:a = aa[501]; break;
    case 18:a = aa[544]; break;
    case 19:a = aa[548]; break;
    case 20:a = aa[565]; break;
    case 21:a = aa[603]; break;
    case 22:a = aa[604]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data14(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[3  ]; break;
    case 1: a = aa[10 ]; break;
    case 2: a = aa[17 ]; break;
    case 3: a = aa[35 ]; break;
    case 4: a = aa[66 ]; break;
    case 5: a = aa[99 ]; break;
    case 6: a = aa[152]; break;
    case 7: a = aa[178]; break;
    case 8: a = aa[248]; break;
    case 9: a = aa[259]; break;
    case 10:a = aa[270]; break;
    case 11:a = aa[290]; break;
    case 12:a = aa[321]; break;
    case 13:a = aa[336]; break;
    case 14:a = aa[337]; break;
    case 15:a = aa[361]; break;
    case 16:a = aa[382]; break;
    case 17:a = aa[393]; break;
    case 18:a = aa[416]; break;
    case 19:a = aa[473]; break;
    case 20:a = aa[502]; break;
    case 21:a = aa[545]; break;
    case 22:a = aa[600]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data15(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[8  ]; break;
    case 1: a = aa[25 ]; break;
    case 2: a = aa[30 ]; break;
    case 3: a = aa[57 ]; break;
    case 4: a = aa[120]; break;
    case 5: a = aa[123]; break;
    case 6: a = aa[169]; break;
    case 7: a = aa[192]; break;
    case 8: a = aa[217]; break;
    case 9: a = aa[241]; break;
    case 10:a = aa[271]; break;
    case 11:a = aa[274]; break;
    case 12:a = aa[285]; break;
    case 13:a = aa[306]; break;
    case 14:a = aa[327]; break;
    case 15:a = aa[368]; break;
    case 16:a = aa[403]; break;
    case 17:a = aa[434]; break;
    case 18:a = aa[474]; break;
    case 19:a = aa[476]; break;
    case 20:a = aa[504]; break;
    case 21:a = aa[538]; break;
    case 22:a = aa[563]; break;
    case 23:a = aa[568]; break;
    case 24:a = aa[596]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data16(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[12 ]; break;
    case 1: a = aa[26 ]; break;
    case 2: a = aa[49 ]; break;
    case 3: a = aa[68 ]; break;
    case 4: a = aa[83 ]; break;
    case 5: a = aa[121]; break;
    case 6: a = aa[219]; break;
    case 7: a = aa[234]; break;
    case 8: a = aa[252]; break;
    case 9: a = aa[265]; break;
    case 10:a = aa[281]; break;
    case 11:a = aa[282]; break;
    case 12:a = aa[300]; break;
    case 13:a = aa[313]; break;
    case 14:a = aa[342]; break;
    case 15:a = aa[378]; break;
    case 16:a = aa[389]; break;
    case 17:a = aa[483]; break;
    case 18:a = aa[496]; break;
    case 19:a = aa[516]; break;
    case 20:a = aa[578]; break;
    case 21:a = aa[582]; break;
    case 22:a = aa[595]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data17(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[11 ]; break;
    case 1: a = aa[67 ]; break;
    case 2: a = aa[77 ]; break;
    case 3: a = aa[104]; break;
    case 4: a = aa[125]; break;
    case 5: a = aa[160]; break;
    case 6: a = aa[203]; break;
    case 7: a = aa[207]; break;
    case 8: a = aa[243]; break;
    case 9: a = aa[244]; break;
    case 10:a = aa[264]; break;
    case 11:a = aa[299]; break;
    case 12:a = aa[323]; break;
    case 13:a = aa[367]; break;
    case 14:a = aa[400]; break;
    case 15:a = aa[401]; break;
    case 16:a = aa[441]; break;
    case 17:a = aa[456]; break;
    case 18:a = aa[480]; break;
    case 19:a = aa[528]; break;
    case 20:a = aa[579]; break;
    case 21:a = aa[589]; break;
    case 22:a = aa[619]; break;
    default: a = 0; break;
  }; 
  return a;
}


uint18_t get_data18(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;
  switch (offset)
  {
    case 0: a = aa[21 ]; break;
    case 1: a = aa[43 ]; break;
    case 2: a = aa[62 ]; break;
    case 3: a = aa[144]; break;
    case 4: a = aa[145]; break;
    case 5: a = aa[196]; break;
    case 6: a = aa[197]; break;
    case 7: a = aa[199]; break;
    case 8: a = aa[292]; break;
    case 9: a = aa[301]; break;
    case 10:a = aa[317]; break;
    case 11:a = aa[330]; break;
    case 12:a = aa[331]; break;
    case 13:a = aa[332]; break;
    case 14:a = aa[350]; break;
    case 15:a = aa[363]; break;
    case 16:a = aa[381]; break;
    case 17:a = aa[433]; break;
    case 18:a = aa[469]; break;
    case 19:a = aa[484]; break;
    case 20:a = aa[522]; break;
    case 21:a = aa[561]; break;
    case 22:a = aa[587]; break;
    case 23:a = aa[623]; break;
    case 24:a = aa[624]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data19(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[16 ]; break;
    case 1: a = aa[60 ]; break;
    case 2: a = aa[69 ]; break;
    case 3: a = aa[80 ]; break;
    case 4: a = aa[112]; break;
    case 5: a = aa[117]; break;
    case 6: a = aa[170]; break;
    case 7: a = aa[186]; break;
    case 8: a = aa[206]; break;
    case 9: a = aa[223]; break;
    case 10:a = aa[255]; break;
    case 11:a = aa[288]; break;
    case 12:a = aa[289]; break;
    case 13:a = aa[325]; break;
    case 14:a = aa[326]; break;
    case 15:a = aa[345]; break;
    case 16:a = aa[357]; break;
    case 17:a = aa[372]; break;
    case 18:a = aa[487]; break;
    case 19:a = aa[508]; break;
    case 20:a = aa[521]; break;
    case 21:a = aa[543]; break;
    case 22:a = aa[581]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data20(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[2  ]; break;
    case 1: a = aa[13 ]; break;
    case 2: a = aa[44 ]; break;
    case 3: a = aa[63 ]; break;
    case 4: a = aa[90 ]; break;
    case 5: a = aa[111]; break;
    case 6: a = aa[126]; break;
    case 7: a = aa[154]; break;
    case 8: a = aa[181]; break;
    case 9: a = aa[200]; break;
    case 10:a = aa[230]; break;
    case 11:a = aa[257]; break;
    case 12:a = aa[294]; break;
    case 13:a = aa[295]; break;
    case 14:a = aa[296]; break;
    case 15:a = aa[339]; break;
    case 16:a = aa[373]; break;
    case 17:a = aa[399]; break;
    case 18:a = aa[422]; break;
    case 19:a = aa[436]; break;
    case 20:a = aa[462]; break;
    case 21:a = aa[518]; break;
    case 22:a = aa[533]; break;
    case 23:a = aa[585]; break;
    case 24:a = aa[610]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data21(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[4  ]; break;
    case 1: a = aa[32 ]; break;
    case 2: a = aa[73 ]; break;
    case 3: a = aa[81 ]; break;
    case 4: a = aa[108]; break;
    case 5: a = aa[172]; break;
    case 6: a = aa[190]; break;
    case 7: a = aa[194]; break;
    case 8: a = aa[224]; break;
    case 9: a = aa[266]; break;
    case 10:a = aa[318]; break;
    case 11:a = aa[338]; break;
    case 12:a = aa[360]; break;
    case 13:a = aa[392]; break;
    case 14:a = aa[437]; break;
    case 15:a = aa[475]; break;
    case 16:a = aa[505]; break;
    case 17:a = aa[574]; break;
    case 18:a = aa[601]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data22(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[71 ]; break;
    case 1: a = aa[85 ]; break;
    case 2: a = aa[92 ]; break;
    case 3: a = aa[124]; break;
    case 4: a = aa[133]; break;
    case 5: a = aa[143]; break;
    case 6: a = aa[166]; break;
    case 7: a = aa[211]; break;
    case 8: a = aa[225]; break;
    case 9: a = aa[304]; break;
    case 10:a = aa[305]; break;
    case 11:a = aa[351]; break;
    case 12:a = aa[352]; break;
    case 13:a = aa[407]; break;
    case 14:a = aa[423]; break;
    case 15:a = aa[431]; break;
    case 16:a = aa[472]; break;
    case 17:a = aa[495]; break;
    case 18:a = aa[515]; break;
    case 19:a = aa[549]; break;
    case 20:a = aa[553]; break;
    case 21:a = aa[558]; break;
    case 22:a = aa[588]; break;
    case 23:a = aa[614]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data23(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[37 ]; break;
    case 1: a = aa[50 ]; break;
    case 2: a = aa[88 ]; break;
    case 3: a = aa[114]; break;
    case 4: a = aa[134]; break;
    case 5: a = aa[189]; break;
    case 6: a = aa[205]; break;
    case 7: a = aa[214]; break;
    case 8: a = aa[236]; break;
    case 9: a = aa[273]; break;
    case 10:a = aa[297]; break;
    case 11:a = aa[349]; break;
    case 12:a = aa[354]; break;
    case 13:a = aa[432]; break;
    case 14:a = aa[457]; break;
    case 15:a = aa[477]; break;
    case 16:a = aa[498]; break;
    case 17:a = aa[512]; break;
    case 18:a = aa[605]; break;
    case 19:a = aa[616]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data24(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[31 ]; break;
    case 1: a = aa[84 ]; break;
    case 2: a = aa[113]; break;
    case 3: a = aa[116]; break;
    case 4: a = aa[129]; break;
    case 5: a = aa[158]; break;
    case 6: a = aa[182]; break;
    case 7: a = aa[227]; break;
    case 8: a = aa[276]; break;
    case 9: a = aa[380]; break;
    case 10:a = aa[404]; break;
    case 11:a = aa[460]; break;
    case 12:a = aa[470]; break;
    case 13:a = aa[493]; break;
    case 14:a = aa[494]; break;
    case 15:a = aa[503]; break;
    case 16:a = aa[514]; break;
    case 17:a = aa[566]; break;
    case 18:a = aa[580]; break;
    case 19:a = aa[602]; break;
    case 20:a = aa[617]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data25(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[33 ]; break;
    case 1: a = aa[51 ]; break;
    case 2: a = aa[64 ]; break;
    case 3: a = aa[78 ]; break;
    case 4: a = aa[86 ]; break;
    case 5: a = aa[110]; break;
    case 6: a = aa[130]; break;
    case 7: a = aa[216]; break;
    case 8: a = aa[254]; break;
    case 9: a = aa[298]; break;
    case 10:a = aa[320]; break;
    case 11:a = aa[402]; break;
    case 12:a = aa[419]; break;
    case 13:a = aa[438]; break;
    case 14:a = aa[446]; break;
    case 15:a = aa[455]; break;
    case 16:a = aa[491]; break;
    case 17:a = aa[500]; break;
    case 18:a = aa[590]; break;
    case 19:a = aa[622]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data26(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[70 ]; break;
    case 1: a = aa[89 ]; break;
    case 2: a = aa[115]; break;
    case 3: a = aa[127]; break;
    case 4: a = aa[142]; break;
    case 5: a = aa[272]; break;
    case 6: a = aa[348]; break;
    case 7: a = aa[370]; break;
    case 8: a = aa[379]; break;
    case 9: a = aa[430]; break;
    case 10:a = aa[461]; break;
    case 11:a = aa[485]; break;
    case 12:a = aa[513]; break;
    case 13:a = aa[541]; break;
    case 14:a = aa[550]; break;
    case 15:a = aa[583]; break;
    case 16:a = aa[607]; break;
    default: a = 0; break;
  }; 
  return a;
}

uint18_t get_data27(int offset, uint18_t aa[ROWS*COLS])
{
  #pragma HLS inline
  uint18_t a;

  switch (offset)
  {
    case 0: a = aa[36 ]; break;
    case 1: a = aa[91 ]; break;
    case 2: a = aa[98 ]; break;
    case 3: a = aa[107]; break;
    case 4: a = aa[176]; break;
    case 5: a = aa[202]; break;
    case 6: a = aa[278]; break;
    case 7: a = aa[467]; break;
    case 8: a = aa[482]; break;
    case 9: a = aa[546]; break;
    case 10:a = aa[556]; break;
    case 11:a = aa[569]; break;
    case 12:a = aa[592]; break;
    case 13:a = aa[620]; break;
    default: a = 0; break;
  }; return a;
}

void get_all_data(uint18_t output[12], uint10_t addr_list[12], uint18_t aa[ROWS*COLS])
{
  
  #pragma HLS inline
  #pragma HLS interface ap_stable port=aa

  uint5_t bank[12];
  #pragma HLS array_partition variable=bank complete dim=0
  uint5_t offset[12];
  #pragma HLS array_partition variable=offset complete dim=0
  uint5_t offset_for_banks[28];
  #pragma HLS array_partition variable=offset_for_banks complete dim=0
  uint18_t data_from_banks[28];
  #pragma HLS array_partition variable=data_from_banks complete dim=0
  
COMPUTE_BANK_AND_OFFSET:
  for (int i = 0; i < 12; i ++ )
  {
    #pragma HLS unroll   
    bank[i] = get_bank(addr_list[i]);
    offset[i] = get_offset(addr_list[i]);
  }

ASSIGN_OFFSET_FOR_BANKS:
  for (int i = 0; i < 28; i ++ )
  {
    #pragma HLS unroll
    offset_for_banks[i] = (bank[ 0] == i) ? offset[ 0] :
                          (bank[ 1] == i) ? offset[ 1] :
                          (bank[ 2] == i) ? offset[ 2] :
                          (bank[ 3] == i) ? offset[ 3] :
                          (bank[ 4] == i) ? offset[ 4] :
                          (bank[ 5] == i) ? offset[ 5] :
                          (bank[ 6] == i) ? offset[ 6] :
                          (bank[ 7] == i) ? offset[ 7] :
                          (bank[ 8] == i) ? offset[ 8] :
                          (bank[ 9] == i) ? offset[ 9] :
                          (bank[10] == i) ? offset[10] :
                          (bank[11] == i) ? offset[11] :
                          uint5_t("0",10);

  }

READ_ALL_BANKS:
  data_from_banks[ 0] =  get_data0 (offset_for_banks[ 0], aa);
  data_from_banks[ 1] =  get_data1 (offset_for_banks[ 1], aa);
  data_from_banks[ 2] =  get_data2 (offset_for_banks[ 2], aa);
  data_from_banks[ 3] =  get_data3 (offset_for_banks[ 3], aa);
  data_from_banks[ 4] =  get_data4 (offset_for_banks[ 4], aa);
  data_from_banks[ 5] =  get_data5 (offset_for_banks[ 5], aa);
  data_from_banks[ 6] =  get_data6 (offset_for_banks[ 6], aa);
  data_from_banks[ 7] =  get_data7 (offset_for_banks[ 7], aa);
  data_from_banks[ 8] =  get_data8 (offset_for_banks[ 8], aa);
  data_from_banks[ 9] =  get_data9 (offset_for_banks[ 9], aa);
  data_from_banks[10] =  get_data10(offset_for_banks[10], aa);
  data_from_banks[11] =  get_data11(offset_for_banks[11], aa);
  data_from_banks[12] =  get_data12(offset_for_banks[12], aa);
  data_from_banks[13] =  get_data13(offset_for_banks[13], aa);
  data_from_banks[14] =  get_data14(offset_for_banks[14], aa);
  data_from_banks[15] =  get_data15(offset_for_banks[15], aa);
  data_from_banks[16] =  get_data16(offset_for_banks[16], aa);
  data_from_banks[17] =  get_data17(offset_for_banks[17], aa);
  data_from_banks[18] =  get_data18(offset_for_banks[18], aa);
  data_from_banks[19] =  get_data19(offset_for_banks[19], aa);
  data_from_banks[20] =  get_data20(offset_for_banks[20], aa);
  data_from_banks[21] =  get_data21(offset_for_banks[21], aa);
  data_from_banks[22] =  get_data22(offset_for_banks[22], aa);
  data_from_banks[23] =  get_data23(offset_for_banks[23], aa);
  data_from_banks[24] =  get_data24(offset_for_banks[24], aa);
  data_from_banks[25] =  get_data25(offset_for_banks[25], aa);
  data_from_banks[26] =  get_data26(offset_for_banks[26], aa);
  data_from_banks[27] =  get_data27(offset_for_banks[27], aa);

CHOOSE_DATA:
  for (int i = 0; i < 12; i ++ )
  {
    #pragma HLS unroll
    output[i] = data_from_banks[bank[i]];
  }

}

int classifier0( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[4][6];
  coord[1] = II[4][18];
  coord[2] = II[13][6];
  coord[3] = II[13][18];

  coord[4] = II[7][6];
  coord[5] = II[7][18];
  coord[6] = II[10][6];
  coord[7] = II[10][18];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-129 * stddev))
     return_value = -567;
  else
     return_value = 534;
  return return_value;
}
int classifier1( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[4][6];
  coord[1] = II[4][18];
  coord[2] = II[11][6];
  coord[3] = II[11][18];

  coord[4] = II[4][10];
  coord[5] = II[4][14];
  coord[6] = II[11][10];
  coord[7] = II[11][14];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (50 * stddev))
     return_value = 339;
  else
     return_value = -477;
  return return_value;
}
int classifier2( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[9][3];
  coord[1] = II[9][21];
  coord[2] = II[18][3];
  coord[3] = II[18][21];

  coord[4] = II[12][3];
  coord[5] = II[12][21];
  coord[6] = II[15][3];
  coord[7] = II[15][21];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (89 * stddev))
     return_value = 272;
  else
     return_value = -386;
  return return_value;
}
int classifier3( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[18][8];
  coord[1] = II[18][17];
  coord[2] = II[24][8];
  coord[3] = II[24][17];

  coord[4] = II[20][8];
  coord[5] = II[20][17];
  coord[6] = II[22][8];
  coord[7] = II[22][17];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (23 * stddev))
     return_value = 301;
  else
     return_value = -223;
  return return_value;
}
int classifier4( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[5][3];
  coord[1] = II[5][7];
  coord[2] = II[24][3];
  coord[3] = II[24][7];

  coord[4] = II[5][5];
  coord[5] = II[5][7];
  coord[6] = II[24][5];
  coord[7] = II[24][7];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (61 * stddev))
     return_value = 322;
  else
     return_value = -199;
  return return_value;
}
int classifier5( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[5][6];
  coord[1] = II[5][18];
  coord[2] = II[21][6];
  coord[3] = II[21][18];

  coord[4] = II[13][6];
  coord[5] = II[13][18];
  coord[6] = II[21][6];
  coord[7] = II[21][18];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (407 * stddev))
     return_value = -479;
  else
     return_value = 142;
  return return_value;
}
int classifier6( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[8][5];
  coord[1] = II[8][17];
  coord[2] = II[14][5];
  coord[3] = II[14][17];

  coord[4] = II[11][5];
  coord[5] = II[11][17];
  coord[6] = II[14][5];
  coord[7] = II[14][17];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (11 * stddev))
     return_value = 112;
  else
     return_value = -432;
  return return_value;
}
int classifier7( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[14][11];
  coord[1] = II[14][15];
  coord[2] = II[24][11];
  coord[3] = II[24][15];

  coord[4] = II[19][11];
  coord[5] = II[19][15];
  coord[6] = II[24][11];
  coord[7] = II[24][15];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-77 * stddev))
     return_value = 113;
  else
     return_value = -378;
  return return_value;
}
int classifier8( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[0][4];
  coord[1] = II[0][11];
  coord[2] = II[6][4];
  coord[3] = II[6][11];

  coord[4] = II[3][4];
  coord[5] = II[3][11];
  coord[6] = II[6][4];
  coord[7] = II[6][11];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (24 * stddev))
     return_value = 218;
  else
     return_value = -219;
  return return_value;
}

int classifier9( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[6][6];
  coord[1] = II[6][18];
  coord[2] = II[12][6];
  coord[3] = II[12][18];

  coord[4] = II[8][6];
  coord[5] = II[8][18];
  coord[6] = II[10][6];
  coord[7] = II[10][18];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-86 * stddev))
     return_value = -402;
  else
     return_value = 318;
  return return_value;
}
int classifier10( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[4][6];
  coord[1] = II[4][18];
  coord[2] = II[11][6];
  coord[3] = II[11][18];

  coord[4] = II[4][10];
  coord[5] = II[4][14];
  coord[6] = II[11][10];
  coord[7] = II[11][14];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (83 * stddev))
     return_value = 302;
  else
     return_value = -414;
  return return_value;
}
int classifier11( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[8][1];
  coord[1] = II[8][20];
  coord[2] = II[20][1];
  coord[3] = II[20][20];

  coord[4] = II[12][1];
  coord[5] = II[12][20];
  coord[6] = II[16][1];
  coord[7] = II[16][20];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (87 * stddev))
     return_value = 179;
  else
     return_value = -497;
  return return_value;
}
int classifier12( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[2][0];
  coord[1] = II[2][24];
  coord[2] = II[5][0];
  coord[3] = II[5][24];

  coord[4] = II[2][8];
  coord[5] = II[2][16];
  coord[6] = II[5][8];
  coord[7] = II[5][16];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (375 * stddev))
     return_value = 442;
  else
     return_value = -142;
  return return_value;
}
int classifier13( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[9][9];
  coord[1] = II[9][15];
  coord[2] = II[24][9];
  coord[3] = II[24][15];

  coord[4] = II[14][9];
  coord[5] = II[14][15];
  coord[6] = II[19][9];
  coord[7] = II[19][15];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (148 * stddev))
     return_value = -558;
  else
     return_value = 68;
  return return_value;
}
int classifier14( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[6][5];
  coord[1] = II[6][19];
  coord[2] = II[16][5];
  coord[3] = II[16][19];

  coord[4] = II[11][5];
  coord[5] = II[11][19];
  coord[6] = II[16][5];
  coord[7] = II[16][19];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-78 * stddev))
     return_value = 116;
  else
     return_value = -684;
  return return_value;
}
int classifier15( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[0][5];
  coord[1] = II[0][19];
  coord[2] = II[9][5];
  coord[3] = II[9][19];

  coord[4] = II[3][5];
  coord[5] = II[3][19];
  coord[6] = II[6][5];
  coord[7] = II[6][19];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (33 * stddev))
     return_value = 137;
  else
     return_value = -277;
  return return_value;
}
int classifier16( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[11][13];
  coord[1] = II[11][22];
  coord[2] = II[17][13];
  coord[3] = II[17][22];

  coord[4] = II[11][16];
  coord[5] = II[11][19];
  coord[6] = II[17][16];
  coord[7] = II[17][19];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (75 * stddev))
     return_value = 238;
  else
     return_value = -90;
  return return_value;
}
int classifier17( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[5][7];
  coord[1] = II[5][13];
  coord[2] = II[15][7];
  coord[3] = II[15][13];

  coord[4] = II[5][9];
  coord[5] = II[5][11];
  coord[6] = II[15][9];
  coord[7] = II[15][11];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-28 * stddev))
     return_value = -169;
  else
     return_value = 237;
  return return_value;
}
int classifier18( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[8][10];
  coord[1] = II[8][16];
  coord[2] = II[18][10];
  coord[3] = II[18][16];

  coord[4] = II[8][12];
  coord[5] = II[8][14];
  coord[6] = II[18][12];
  coord[7] = II[18][14];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-40 * stddev))
     return_value = -76;
  else
     return_value = 296;
  return return_value;
}
int classifier19( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[5][2];
  coord[1] = II[5][6];
  coord[2] = II[14][2];
  coord[3] = II[14][6];

  coord[4] = II[5][4];
  coord[5] = II[5][6];
  coord[6] = II[14][4];
  coord[7] = II[14][6];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (64 * stddev))
     return_value = 347;
  else
     return_value = -107;
  return return_value;
}
int classifier20( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[0][18];
  coord[1] = II[0][24];
  coord[2] = II[11][18];
  coord[3] = II[11][24];

  coord[4] = II[0][20];
  coord[5] = II[0][22];
  coord[6] = II[11][20];
  coord[7] = II[11][22];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-84 * stddev))
     return_value = -50;
  else
     return_value = 373;
  return return_value;
}
int classifier21( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[6][0];
  coord[1] = II[6][24];
  coord[2] = II[19][0];
  coord[3] = II[19][24];

  coord[4] = II[6][8];
  coord[5] = II[6][16];
  coord[6] = II[19][8];
  coord[7] = II[19][16];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-563 * stddev))
     return_value = -135;
  else
     return_value = 286;
  return return_value;
}
int classifier22( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[6][9];
  coord[1] = II[6][15];
  coord[2] = II[15][9];
  coord[3] = II[15][15];

  coord[4] = II[6][11];
  coord[5] = II[6][13];
  coord[6] = II[15][11];
  coord[7] = II[15][13];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (58 * stddev))
     return_value = 292;
  else
     return_value = -89;
  return return_value;
}
int classifier23( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[18][7];
  coord[1] = II[18][17];
  coord[2] = II[24][7];
  coord[3] = II[24][17];

  coord[4] = II[20][7];
  coord[5] = II[20][17];
  coord[6] = II[22][7];
  coord[7] = II[22][17];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (41 * stddev))
     return_value = 197;
  else
     return_value = -155;
  return return_value;
}
int classifier24( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[7][5];
  coord[1] = II[7][19];
  coord[2] = II[19][5];
  coord[3] = II[19][19];

  coord[4] = II[13][5];
  coord[5] = II[13][19];
  coord[6] = II[19][5];
  coord[7] = II[19][19];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (374 * stddev))
     return_value = -387;
  else
     return_value = 99;
  return return_value;
}
int classifier25( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[3][0];
  coord[1] = II[3][24];
  coord[2] = II[6][0];
  coord[3] = II[6][24];

  coord[4] = II[3][8];
  coord[5] = II[3][16];
  coord[6] = II[6][8];
  coord[7] = II[6][16];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (285 * stddev))
     return_value = 375;
  else
     return_value = -259;
  return return_value;
}
int classifier26( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[8][5];
  coord[1] = II[8][20];
  coord[2] = II[14][5];
  coord[3] = II[14][20];

  coord[4] = II[11][5];
  coord[5] = II[11][20];
  coord[6] = II[14][5];
  coord[7] = II[14][20];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (129 * stddev))
     return_value = 256;
  else
     return_value = -421;
  return return_value;
}
int classifier27( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[6][9];
  coord[1] = II[6][14];
  coord[2] = II[20][9];
  coord[3] = II[20][14];

  coord[4] = II[13][9];
  coord[5] = II[13][14];
  coord[6] = II[20][9];
  coord[7] = II[20][14];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (58 * stddev))
     return_value = -408;
  else
     return_value = 118;
  return return_value;
}
int classifier28( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[5][9];
  coord[1] = II[5][15];
  coord[2] = II[15][9];
  coord[3] = II[15][15];

  coord[4] = II[5][11];
  coord[5] = II[5][13];
  coord[6] = II[15][11];
  coord[7] = II[15][13];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (59 * stddev))
     return_value = 212;
  else
     return_value = -167;
  return return_value;
}
int classifier29( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[6][6];
  coord[1] = II[6][9];
  coord[2] = II[18][6];
  coord[3] = II[18][9];

  coord[4] = II[12][6];
  coord[5] = II[12][9];
  coord[6] = II[18][6];
  coord[7] = II[18][9];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-12 * stddev))
     return_value = 108;
  else
     return_value = -357;
  return return_value;
}
int classifier30( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[21][3];
  coord[1] = II[21][21];
  coord[2] = II[24][3];
  coord[3] = II[24][21];

  coord[4] = II[21][9];
  coord[5] = II[21][15];
  coord[6] = II[24][9];
  coord[7] = II[24][15];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (134 * stddev))
     return_value = 269;
  else
     return_value = -129;
  return return_value;
}
int classifier31( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[6][5];
  coord[1] = II[6][18];
  coord[2] = II[12][5];
  coord[3] = II[12][18];

  coord[4] = II[8][5];
  coord[5] = II[8][18];
  coord[6] = II[10][5];
  coord[7] = II[10][18];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-29 * stddev))
     return_value = -344;
  else
     return_value = 93;
  return return_value;
}
int classifier32( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[1][18];
  coord[1] = II[1][24];
  coord[2] = II[16][18];
  coord[3] = II[16][24];

  coord[4] = II[1][18];
  coord[5] = II[1][21];
  coord[6] = II[16][18];
  coord[7] = II[16][21];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (206 * stddev))
     return_value = 371;
  else
     return_value = -77;
  return return_value;
}
int classifier33( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[1][1];
  coord[1] = II[1][7];
  coord[2] = II[16][1];
  coord[3] = II[16][7];

  coord[4] = II[1][4];
  coord[5] = II[1][7];
  coord[6] = II[16][4];
  coord[7] = II[16][7];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (192 * stddev))
     return_value = 310;
  else
     return_value = -103;
  return return_value;
}
int classifier34( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[8][0];
  coord[1] = II[8][24];
  coord[2] = II[23][0];
  coord[3] = II[23][24];

  coord[4] = II[8][8];
  coord[5] = II[8][16];
  coord[6] = II[23][8];
  coord[7] = II[23][16];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-284 * stddev))
     return_value = -117;
  else
     return_value = 269;
  return return_value;
}
int classifier35( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[6][5];
  coord[1] = II[6][19];
  coord[2] = II[18][5];
  coord[3] = II[18][19];

  coord[4] = II[6][5];
  coord[5] = II[6][12];
  coord[6] = II[12][5];
  coord[7] = II[12][12];

  coord[8] = II[12][12];
  coord[9] = II[12][19];
  coord[10] = II[18][12];
  coord[11] = II[18][19];

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 8192;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-200 * stddev))
     return_value = 39;
  else
     return_value = -416;
  return return_value;
}
int classifier36( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[12][2];
  coord[1] = II[12][23];
  coord[2] = II[24][2];
  coord[3] = II[24][23];

  coord[4] = II[16][2];
  coord[5] = II[16][23];
  coord[6] = II[20][2];
  coord[7] = II[20][23];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (347 * stddev))
     return_value = -400;
  else
     return_value = 72;
  return return_value;
}
int classifier37( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[1][8];
  coord[1] = II[1][12];
  coord[2] = II[11][8];
  coord[3] = II[11][12];

  coord[4] = II[1][10];
  coord[5] = II[1][12];
  coord[6] = II[11][10];
  coord[7] = II[11][12];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-7 * stddev))
     return_value = 59;
  else
     return_value = -259;
  return return_value;
}
int classifier38( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[13][2];
  coord[1] = II[13][22];
  coord[2] = II[23][2];
  coord[3] = II[23][22];

  coord[4] = II[13][2];
  coord[5] = II[13][12];
  coord[6] = II[23][2];
  coord[7] = II[23][12];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (473 * stddev))
     return_value = 327;
  else
     return_value = -42;
  return return_value;
}
int classifier39( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[1][0];
  coord[1] = II[1][6];
  coord[2] = II[14][0];
  coord[3] = II[14][6];

  coord[4] = II[1][2];
  coord[5] = II[1][4];
  coord[6] = II[14][2];
  coord[7] = II[14][4];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-210 * stddev))
     return_value = -77;
  else
     return_value = 388;
  return return_value;
}
int classifier40( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[2][20];
  coord[1] = II[2][24];
  coord[2] = II[15][20];
  coord[3] = II[15][24];

  coord[4] = II[2][20];
  coord[5] = II[2][22];
  coord[6] = II[15][20];
  coord[7] = II[15][22];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-174 * stddev))
     return_value = -13;
  else
     return_value = 451;
  return return_value;
}
int classifier41( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[5][0];
  coord[1] = II[5][22];
  coord[2] = II[24][0];
  coord[3] = II[24][22];

  coord[4] = II[5][11];
  coord[5] = II[5][22];
  coord[6] = II[24][11];
  coord[7] = II[24][22];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (1522 * stddev))
     return_value = 393;
  else
     return_value = -80;
  return return_value;
}
int classifier42( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[4][18];
  coord[1] = II[4][24];
  coord[2] = II[13][18];
  coord[3] = II[13][24];

  coord[4] = II[4][20];
  coord[5] = II[4][22];
  coord[6] = II[13][20];
  coord[7] = II[13][22];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (79 * stddev))
     return_value = 239;
  else
     return_value = -25;
  return return_value;
}
int classifier43( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[3][0];
  coord[1] = II[3][6];
  coord[2] = II[14][0];
  coord[3] = II[14][6];

  coord[4] = II[3][2];
  coord[5] = II[3][4];
  coord[6] = II[14][2];
  coord[7] = II[14][4];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (71 * stddev))
     return_value = 246;
  else
     return_value = -103;
  return return_value;
}
int classifier44( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[1][12];
  coord[1] = II[1][16];
  coord[2] = II[10][12];
  coord[3] = II[10][16];

  coord[4] = II[1][12];
  coord[5] = II[1][14];
  coord[6] = II[10][12];
  coord[7] = II[10][14];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (162 * stddev))
     return_value = -757;
  else
     return_value = 43;
  return return_value;
}
int classifier45( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[6][0];
  coord[1] = II[6][19];
  coord[2] = II[9][0];
  coord[3] = II[9][19];

  coord[4] = II[7][0];
  coord[5] = II[7][19];
  coord[6] = II[8][0];
  coord[7] = II[8][19];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-37 * stddev))
     return_value = -112;
  else
     return_value = 227;
  return return_value;
}
int classifier46( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[1][12];
  coord[1] = II[1][16];
  coord[2] = II[10][12];
  coord[3] = II[10][16];

  coord[4] = II[1][12];
  coord[5] = II[1][14];
  coord[6] = II[10][12];
  coord[7] = II[10][14];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (7 * stddev))
     return_value = 102;
  else
     return_value = -95;
  return return_value;
}
int classifier47( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[1][8];
  coord[1] = II[1][12];
  coord[2] = II[10][8];
  coord[3] = II[10][12];

  coord[4] = II[1][10];
  coord[5] = II[1][12];
  coord[6] = II[10][10];
  coord[7] = II[10][12];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (123 * stddev))
     return_value = -677;
  else
     return_value = 16;
  return return_value;
}
int classifier48( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[5][5];
  coord[1] = II[5][19];
  coord[2] = II[19][5];
  coord[3] = II[19][19];

  coord[4] = II[5][12];
  coord[5] = II[5][19];
  coord[6] = II[12][12];
  coord[7] = II[12][19];

  coord[8] = II[12][5];
  coord[9] = II[12][12];
  coord[10] = II[19][5];
  coord[11] = II[19][12];

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 8192;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-322 * stddev))
     return_value = 72;
  else
     return_value = -447;
  return return_value;
}
int classifier49( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[10][1];
  coord[1] = II[10][19];
  coord[2] = II[12][1];
  coord[3] = II[12][19];

  coord[4] = II[11][1];
  coord[5] = II[11][19];
  coord[6] = II[12][1];
  coord[7] = II[12][19];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (8 * stddev))
     return_value = 59;
  else
     return_value = -240;
  return return_value;
}
int classifier50( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[13][17];
  coord[1] = II[13][21];
  coord[2] = II[24][17];
  coord[3] = II[24][21];

  coord[4] = II[13][17];
  coord[5] = II[13][19];
  coord[6] = II[24][17];
  coord[7] = II[24][19];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 8192;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (110 * stddev))
     return_value = 275;
  else
     return_value = -13;
  return return_value;
}
int classifier51( int_II II[WINDOW_SIZE][WINDOW_SIZE], int stddev ){
  static int coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  int sum0, sum1, sum2, final_sum;
  int return_value;
  coord[0] = II[4][0];
  coord[1] = II[4][6];
  coord[2] = II[13][0];
  coord[3] = II[13][6];

  coord[4] = II[7][0];
  coord[5] = II[7][6];
  coord[6] = II[10][0];
  coord[7] = II[10][6];

  coord[8] = 0;
  coord[9] = 0;
  coord[10] = 0;
  coord[11] = 0;

  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * -4096;
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * 12288;
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * 0;
  final_sum = sum0 + sum1 + sum2;

  if(final_sum >= (-184 * stddev))
     return_value = 25;
  else
     return_value = -468;
  return return_value;
}

/****************************************************************************************/
/* FUNCTION DECLARATIONS
****************************************************************************************/
void imageScaler        ( int src_height,
			  int src_width,
                          unsigned char Data[IMAGE_HEIGHT][IMAGE_WIDTH],
                          int dest_height,
			  int dest_width,
                          unsigned char IMG1_data[IMAGE_HEIGHT][IMAGE_WIDTH] 
                        );

void processImage       ( float _factor,
                          int sum_row,
                          int sum_col,
                          int *AllCandidates_x,
                          int *AllCandidates_y,
                          int *AllCandidates_w,
                          int *AllCandidates_h,
                          int *AllCandidates_size,
                          unsigned char IMG1_data[IMAGE_HEIGHT][IMAGE_WIDTH],
                          MySize winSize
                        );

int cascadeClassifier   ( MyPoint pt,
                          int_II II[WINDOW_SIZE][WINDOW_SIZE],
                          int_SII SII[SQ_SIZE][SQ_SIZE] 
                        );

int weakClassifier      ( int stddev,
                          uint18_t coord[12],
                          int haar_counter,
                          int w_id                      
                        );

void groupRectangles    ( MyRect *rectList, 
                          int *rectList_size, 
                          int groupThreshold, 
                          float eps
                        );

unsigned int int_sqrt   ( ap_uint<32> value
                        );

inline  int  myRound ( float value )
{
  return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

//========================================================================================
// TOP LEVEL MODULE OR DUT (DEVICE UNDER TEST) 
//========================================================================================

void face_detect

( 
  unsigned char inData[IMAGE_WIDTH],
  int result_x[RESULT_SIZE],
  int result_y[RESULT_SIZE],
  int result_w[RESULT_SIZE],
  int result_h[RESULT_SIZE],
  int *result_size
)

{
  static unsigned char Data[IMAGE_HEIGHT][IMAGE_WIDTH];
  int i, j;

  int result_x_Scale[RESULT_SIZE];
  int result_y_Scale[RESULT_SIZE];
  int result_w_Scale[RESULT_SIZE];
  int result_h_Scale[RESULT_SIZE];
  int res_size_Scale = 0;
  int *result_size_Scale = &res_size_Scale;

  static int counter = 0;
  if ( counter < IMAGE_HEIGHT){
    for( j = 0; j < IMAGE_WIDTH; j++){
         Data[counter][j] = inData[j];
    }
    counter++;
    if ( counter < IMAGE_HEIGHT ){
      for ( i = 0; i < RESULT_SIZE; i++){ 
        result_x[i] = 0;
        result_y[i] = 0;
        result_w[i] = 0;
        result_h[i] = 0;
      }
      *result_size = 0; 
      return ;
    }
  }

  counter = 0;
  *result_size = 0;
    
  float  scaleFactor = 1.2;

  unsigned char IMG1_data[IMAGE_HEIGHT][IMAGE_WIDTH];  

  int height, width;   
  float factor;

  MySize winSize0;
  winSize0.width = 24;
  winSize0.height= 24;

  factor = scaleFactor ;

  L1: 
  while ( IMAGE_WIDTH/factor > WINDOW_SIZE && IMAGE_HEIGHT/factor > WINDOW_SIZE )
  {

    /* size of the image scaled up */
    MySize winSize = { myRound(winSize0.width*factor), myRound(winSize0.height*factor) };
    /* size of the image scaled down (from bigger to smaller) */
    MySize sz = { (int)( IMAGE_WIDTH/factor ), (int)( IMAGE_HEIGHT/factor ) };

    height = sz.height;
    width  = sz.width;
     
    imageScaler	    ( IMAGE_HEIGHT,
		      IMAGE_WIDTH,
                      Data, 
                      height,
		      width,
                      IMG1_data
                    ); 
   
  
    processImage       ( factor, 
                         height, 
                         width,
                         result_x_Scale,
                         result_y_Scale,
                         result_w_Scale,
                         result_h_Scale,
                         result_size_Scale,
                         IMG1_data,
                         winSize
                       ); 
    factor *= scaleFactor;
  } /* end of the factor loop, finish all scales in pyramid*/

   for ( i = 0; i < RESULT_SIZE; i++){
      result_x[i] = result_x_Scale[i];
      result_y[i] = result_y_Scale[i];
      result_w[i] = result_w_Scale[i];
      result_h[i] = result_h_Scale[i];
   }
   *result_size = *result_size_Scale;
}

void processImage 

( float factor,
  int sum_row,
  int sum_col, 
  int *AllCandidates_x,
  int *AllCandidates_y,
  int *AllCandidates_w,
  int *AllCandidates_h,
  int *AllCandidates_size, 
  unsigned char IMG1_data[IMAGE_HEIGHT][IMAGE_WIDTH], 
  MySize winSize
)
{
  #pragma HLS inline off
  MyPoint p;
  int result;
  int step;
 
  int u,v;
  int x,y,i,j,k;

  /** Image Line buffer ( 24 BRAMs ) */ 
  unsigned char L[WINDOW_SIZE-1][IMAGE_WIDTH];
  #pragma HLS array_partition variable=L complete dim=1

  /** Image Window buffer ( 1250 registers )*/
  static int_I I[WINDOW_SIZE][2*WINDOW_SIZE];
  #pragma HLS array_partition variable=I complete dim=0

  /** Integral Image Window buffer ( 625 registers )*/
  static int_II II[WINDOW_SIZE][WINDOW_SIZE];
  #pragma HLS array_partition variable=II complete dim=0
  
  /** Square Image Window buffer ( 1250 registers ) **/
  static int_SI SI[WINDOW_SIZE][2*WINDOW_SIZE];
  #pragma HLS array_partition variable=SI complete dim=0

  /** Square Integral Image Window buffer ( 625 registers )*/
  static int_SII SII[SQ_SIZE][SQ_SIZE];
  #pragma HLS array_partition variable=SII complete dim=0

  
  Initialize0u : 
  for ( u = 0; u < WINDOW_SIZE; u++ ){
  #pragma HLS unroll
    Initailize0v:
    for ( v = 0; v < WINDOW_SIZE; v++ ){
    #pragma HLS unroll
      II[u][v] = 0;
    }
  }

  SII[0][0] = 0;
  SII[0][1] = 0;
  SII[1][0] = 0;
  SII[1][1] = 0;
  

  Initialize1i:
  for ( i = 0; i < WINDOW_SIZE ; i++ ){
  #pragma HLS unroll
    Initialize1j:
    for ( j = 0; j < 2*WINDOW_SIZE; j++ ){
    #pragma HLS unroll
      I[i][j] = 0;
      SI[i][j] = 0;
    }
  }
  
  
  Initialize2y :
  for ( y = 0; y < WINDOW_SIZE-1; y++ ){
  #pragma HLS unroll
    Initialize2x :
    for ( x = 0; x < IMAGE_WIDTH ; x++){//IMAGE_WIDTH; x++ ){
      L[y][x] = 0;
    }
  }
 

  int element_counter = 0;
  int x_index = 0;
  int y_index = 0;
  int ii, jj;

  /** Loop over each point in the Image ( scaled ) **/
  Pixely: for( y = 0; y < sum_row; y++ ){
    Pixelx : for ( x = 0; x < sum_col; x++ ){

      /* Updates for Integral Image Window Buffer (I) */
      SetIIu: for ( u = 0; u < WINDOW_SIZE; u++){
      #pragma HLS unroll
        SetIIj: for ( v = 0; v < WINDOW_SIZE; v++ ){
        #pragma HLS unroll
          II[u][v] = II[u][v] + ( I[u][v+1] - I[u][0] );
        }
      }
      
      /* Updates for Square Image Window Buffer (SI) */
      SII[0][0] = SII[0][0] + ( SI[0][1] - SI[0][0] );
      SII[0][1] = SII[0][1] + ( SI[0][WINDOW_SIZE] - SI[0][0] );
      SII[1][0] = SII[1][0] + ( SI[WINDOW_SIZE-1][1] - SI[WINDOW_SIZE-1][0] );
      SII[1][1] = SII[1][1] + ( SI[WINDOW_SIZE-1][WINDOW_SIZE] - SI[WINDOW_SIZE-1][0] );
      
      /* Updates for Image Window Buffer (I) and Square Image Window Bufer (SI) */
      SetIj: for( j = 0; j < 2*WINDOW_SIZE-1; j++){
      #pragma HLS unroll
        SetIi: for( i = 0; i < WINDOW_SIZE; i++ ){
        #pragma HLS unroll
          if( i+j != 2*WINDOW_SIZE-1 ){
            I[i][j] = I[i][j+1];
            SI[i][j] = SI[i][j+1];
          }
          else if ( i > 0 ){
            I[i][j] = I[i][j+1] + I[i-1][j+1];
            SI[i][j] = SI[i][j+1] + SI[i-1][j+1];
          }
        }
      }
      /** Last column of the I[][] matrix **/
      Ilast: for( i = 0; i < WINDOW_SIZE-1; i++ ){
      #pragma HLS unroll
        I[i][2*WINDOW_SIZE-1] = L[i][x];
        SI[i][2*WINDOW_SIZE-1] = L[i][x]*L[i][x];
      }
      I[WINDOW_SIZE-1][2*WINDOW_SIZE-1] = IMG1_data[y][x];
      SI[WINDOW_SIZE-1][2*WINDOW_SIZE-1] = IMG1_data[y][x]*IMG1_data[y][x];

      /** Updates for Image Line Buffer (L) **/
      LineBuf: for( k = 0; k < WINDOW_SIZE-2; k++ ){
      #pragma HLS unroll
        L[k][x] = L[k+1][x];
      }
      L[WINDOW_SIZE-2][x] = IMG1_data[y][x];

      /* Pass the Integral Image Window buffer through Cascaded Classifier. Only pass
       * when the integral image window buffer has flushed out the initial garbage data */
      if ( element_counter >= ( ( (WINDOW_SIZE-1)*sum_col + WINDOW_SIZE ) + WINDOW_SIZE -1 ) ) {

	 /* Sliding Window should not go beyond the boundary */
         if ( x_index < ( sum_col - (WINDOW_SIZE-1) ) && y_index < ( sum_row - (WINDOW_SIZE-1) ) ){
            p.x = x_index;
            p.y = y_index;
            
            result = cascadeClassifier ( p,
                                         II,
                                         SII
                                       );

           if ( result > 0 )
           {
             MyRect r = {myRound(p.x*factor), myRound(p.y*factor), winSize.width, winSize.height};
             AllCandidates_x[*AllCandidates_size]=r.x;
             AllCandidates_y[*AllCandidates_size]=r.y;
             AllCandidates_w[*AllCandidates_size]=r.width;
             AllCandidates_h[*AllCandidates_size]=r.height;
            *AllCandidates_size=*AllCandidates_size+1;
           }
         }// inner if
         if ( x_index < sum_col-1 )
             x_index = x_index + 1;
         else{ 
             x_index = 0;
             y_index = y_index + 1;
         }
       } // outer if
       element_counter +=1;
    }   
  } 
   
}

int cascadeClassifier 
(  
  MyPoint pt, 
  int_II II[WINDOW_SIZE][WINDOW_SIZE],
  int_SII SII[SQ_SIZE][SQ_SIZE]
) 

{
  #pragma HLS INLINE

  int i, j, k;

  int mean;
  int stddev = 0;
  int haar_counter = 0;
  int w_index = 0;
  int r_index = 0;
  int stage_sum=0;

  /* The rectangle co-ordinagte values for all the classifiers */
  #include "haar_dataRcc_with_partitioning.h"
 
  static uint18_t coord[12];
  #pragma HLS array_partition variable=coord complete dim=0

  static int s0[9];
  #pragma HLS array_partition variable=s0 complete dim=0

  static int s1[16];
  #pragma HLS array_partition variable=s1 complete dim=0

  static int s2[27];
  #pragma HLS array_partition variable=s2 complete dim=0

  /* Banking */

  /* 12 (x,y,w,h) values corresponding to 3 rectangles that need to be read */
  uint18_t values[12];
  #pragma HLS array_partition variable=values complete dim=0

  /* location/address of those 12 values in the 25 X 25 window */
  uint10_t addr_list[12];
  #pragma HLS array_partition variable=addr_list complete dim=0

  /* among the 12 values which of them are needed to be read from the 25 X 25 window */
  bit enable_list[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  #pragma HLS array_partition variable=enable_list complete dim=0  
  
  uint18_t _II[WINDOW_SIZE*WINDOW_SIZE];
  #pragma HLS array_partition variable=_II complete dim=0

  COPY_LOOP1: for (int i = 0; i < WINDOW_SIZE; i ++ ){
    #pragma HLS unroll
    COPY_LOOP2: for (int j = 0; j < WINDOW_SIZE; j ++ ){
      #pragma HLS unroll
      _II[i*25+j] = II[i][j];
    }
  }

  stddev =                    SII[0][0]
                           -  SII[0][SQ_SIZE- 1]
                           -  SII[SQ_SIZE-1][0] 
                           +  SII[SQ_SIZE-1][SQ_SIZE-1]; 

  mean =                      II[0][0]
                           -  II[0][WINDOW_SIZE - 1]
                           -  II[WINDOW_SIZE - 1][0] 
                           +  II[WINDOW_SIZE - 1][WINDOW_SIZE - 1];
      
  stddev = (stddev*(WINDOW_SIZE-1)*(WINDOW_SIZE-1));
  stddev =  stddev - mean*mean; 
  
  if( stddev > 0 )
    stddev = int_sqrt(stddev);  
  else
    stddev = 1;

  MyRect tr0,tr1,tr2;
  
  int r_id;
  int w_id;
  int s;

  /* Hard-Coding Classifier 0 */
  stage_sum = 0;
  s0[0] = classifier0( II, stddev );
  s0[1] = classifier1( II, stddev );
  s0[2] = classifier2( II, stddev );
  s0[3] = classifier3( II, stddev );
  s0[4] = classifier4( II, stddev );
  s0[5] = classifier5( II, stddev );
  s0[6] = classifier6( II, stddev );
  s0[7] = classifier7( II, stddev );
  s0[8] = classifier8( II, stddev );
  stage_sum = s0[0] + s0[1] + s0[2] + s0[3] + s0[4] + s0[5] + s0[6] + s0[7] + s0[8];

  if( stage_sum < 0.4*stages_thresh_array[0] )
    return -1;

  haar_counter += 9;

  /* Hard-Coding Classifier 1 */
  stage_sum = 0;
  s1[0] = classifier9( II, stddev );
  s1[1] = classifier10( II, stddev );
  s1[2] = classifier11( II, stddev );
  s1[3] = classifier12( II, stddev );
  s1[4] = classifier13( II, stddev );
  s1[5] = classifier14( II, stddev );
  s1[6] = classifier15( II, stddev );
  s1[7] = classifier16( II, stddev );
  s1[8] = classifier17( II, stddev );
  s1[9] = classifier18( II, stddev );
  s1[10] = classifier19( II, stddev );
  s1[11] = classifier20( II, stddev );
  s1[12] = classifier21( II, stddev );
  s1[13] = classifier22( II, stddev );
  s1[14] = classifier23( II, stddev );
  s1[15] = classifier24( II, stddev );

  stage_sum = s1[0] + s1[1] + s1[2] + s1[3] + s1[4] + s1[5] + s1[6] + s1[7];
  stage_sum+= s1[8] + s1[9] + s1[10] + s1[11] + s1[12] + s1[13] + s1[14] + s1[15];

  if( stage_sum < 0.4*stages_thresh_array[1] )
    return -1;
  
  haar_counter += 16;

  /* Hard-Coding Classifier 2 */
  stage_sum = 0;
  s2[0] = classifier25( II, stddev );
  s2[1] = classifier26( II, stddev );
  s2[2] = classifier27( II, stddev );
  s2[3] = classifier28( II, stddev );
  s2[4] = classifier29( II, stddev );
  s2[5] = classifier30( II, stddev );
  s2[6] = classifier31( II, stddev );
  s2[7] = classifier32( II, stddev );
  s2[8] = classifier33( II, stddev );
  s2[9] = classifier34( II, stddev );
  s2[10] = classifier35( II, stddev );
  s2[11] = classifier36( II, stddev );
  s2[12] = classifier37( II, stddev );
  s2[13] = classifier38( II, stddev );
  s2[14] = classifier39( II, stddev );
  s2[15] = classifier40( II, stddev );
  s2[16] = classifier41( II, stddev );
  s2[17] = classifier42( II, stddev );
  s2[18] = classifier43( II, stddev );
  s2[19] = classifier44( II, stddev );
  s2[20] = classifier45( II, stddev );
  s2[21] = classifier46( II, stddev );
  s2[22] = classifier47( II, stddev );
  s2[23] = classifier48( II, stddev );
  s2[24] = classifier49( II, stddev );
  s2[25] = classifier50( II, stddev );
  s2[26] = classifier51( II, stddev );

  stage_sum = s2[0] + s2[1] + s2[2] + s2[3] + s2[4] + s2[5] + s2[6] + s2[7];
  stage_sum+= s2[8] + s2[9] + s2[10] + s2[11] + s2[12] + s2[13] + s2[14] + s2[15];
  stage_sum+= s2[16] + s2[17] + s2[18] + s2[19] + s2[20] + s2[21] + s2[22] + s2[23];
  stage_sum+= s2[24] + s2[25] + s2[26];

  if( stage_sum < 0.4*stages_thresh_array[2] )
    return -1;
  
  haar_counter += 27;

 /******************************************/
 // REST 23 STAGES
 /*****************************************/

  Stages: for ( i = 3; i < 25; i++ ){
    Filters: for ( j = 0; j < stages_array[i] ; j++ ){

      #pragma HLS pipeline 
      if ( j == 0 ) {
        stage_sum = 0; s=0;
      } 
    
      r_id = r_index;
      w_id = w_index;

      tr0.x = rectangles_array0[haar_counter];
      tr0.width = rectangles_array2[haar_counter];
      tr0.y = rectangles_array1[haar_counter];
      tr0.height = rectangles_array3[haar_counter];
   
      tr1.x = rectangles_array4[haar_counter];
      tr1.width = rectangles_array6[haar_counter];
      tr1.y = rectangles_array5[haar_counter];
      tr1.height = rectangles_array7[haar_counter];
    
      tr2.x = rectangles_array8[haar_counter];
      tr2.width = rectangles_array10[haar_counter];
      tr2.y = rectangles_array9[haar_counter];
      tr2.height = rectangles_array11[haar_counter];

      /* Calculates addresses/locations of all the 12 co-ordinates */
      addr_list[ 0] =  tr0.y    *25+ tr0.x;
      addr_list[ 1] =  tr0.y    *25+tr0.x+tr0.width;
      addr_list[ 2] = (tr0.y+tr0.height)*25+tr0.x;
      addr_list[ 3] = (tr0.y+tr0.height)*25+tr0.x+tr0.width;
      addr_list[ 4] =  tr1.y    *25+tr1.x;
      addr_list[ 5] =  tr1.y    *25+tr1.x+tr1.width;
      addr_list[ 6] = (tr1.y+tr1.height)*25+tr1.x;
      addr_list[ 7] = (tr1.y+tr1.height)*25+tr1.x+tr1.width;

     
      if (!(tr2.x ==0 && tr2.width==0 && tr2.y==0 && tr2.height==0 ) && tr2.width!=0 && tr2.height!=0)
      {
       
        addr_list[ 8] =  tr2.y    *25+tr2.x;
        addr_list[ 9] =  tr2.y    *25+tr2.x+tr2.width;
        addr_list[10] = (tr2.y+tr2.height)*25+tr2.x;
        addr_list[11] = (tr2.y+tr2.height)*25+tr2.x+tr2.width;
        enable_list[ 8] = 1;
        enable_list[ 9] = 1;
        enable_list[10] = 1;
        enable_list[11] = 1;

        }
        else
        {
          addr_list[ 8] =  addr_list[0];
          addr_list[ 9] =  addr_list[1];
          addr_list[10] =  addr_list[2];
          addr_list[11] =  addr_list[3];
          enable_list[ 8] = 0;
          enable_list[ 9] = 0;         
          enable_list[10] = 0;
          enable_list[11] = 0;
  
        }
        
        /* Read the values corresponding to the 12 locations from _II which is the copy of integral image window */
        get_all_data(values, addr_list, _II);
 
  	for ( k = 0; k < 12; k++ ){
	  if ( enable_list[k] == 0 )
            coord[k] = 0;
 	  else 
	    coord[k] = values[k];
        }

        s = weakClassifier      ( stddev, 
                                  coord,
                                  haar_counter,
                                  w_id				 
                                );
          
        stage_sum = stage_sum + s;
        haar_counter = haar_counter+1;
        w_index = w_index+3;
        r_index = r_index+12;
     
    } /* end of j loop */
    if( stage_sum < 0.4*stages_thresh_array[i] ){
       return -i;
    }
  } /* end of i loop */

 return 1;
}
 
int weakClassifier
(
  int stddev,
  uint18_t coord[12],  
  int haar_counter,
  int w_id 
)
{
  /* weights and threshold values for various classifiers */
                                                                                             
  #include "haar_dataEWC_with_partitioning.h"
  # pragma HLS INLINE

  int t = tree_thresh_array[haar_counter] * stddev; 
  
  int sum0 =0;
  int sum1 =0;
  int sum2 =0;
  int final_sum =0;
  int return_value;
	             
  sum0 = (coord[0] - coord[1] - coord[2] + coord[3]) * weights_array0[haar_counter];
  sum1 = (coord[4] - coord[5] - coord[6] + coord[7]) * weights_array1[haar_counter];  
  sum2 = (coord[8] - coord[9] - coord[10] + coord[11]) * weights_array2[haar_counter];
  final_sum = sum0+sum1+sum2;
  
  if(final_sum >= t) 
    return_value =  alpha2_array[haar_counter];  
  else   
    return_value =  alpha1_array[haar_counter];
  
  return return_value ; 
}


void imageScaler
(
  int src_height,
  int src_width,
  unsigned char Data[IMAGE_HEIGHT][IMAGE_WIDTH], 
  int dest_height,
  int dest_width,
  unsigned char IMG1_data[IMAGE_HEIGHT][IMAGE_WIDTH]
)
{
  int y;
  int j;
  int x;
  int i;
  unsigned char* t;
  unsigned char* p;
  int w1 = src_width;
  int h1 = src_height;
  int w2 = dest_width;
  int h2 = dest_height;

  int rat = 0;

  int x_ratio = (int)((w1<<16)/w2) +1; 
  int y_ratio = (int)((h1<<16)/h2) +1;


  imageScalerL1: for ( i = 0 ; i < IMAGE_HEIGHT ; i++ ){ 
    imageScalerL1_1: for (j=0;j < IMAGE_WIDTH ;j++){
      #pragma HLS pipeline
      if ( j < w2 && i < h2 ) 
        IMG1_data[i][j] =  Data[(i*y_ratio)>>16][(j*x_ratio)>>16];
       
    }
  }
}

unsigned int int_sqrt 
( 
  ap_uint<32> value
)
{
  int i;
  unsigned int a = 0, b = 0, c = 0;

  for ( i = 0 ; i < (32 >> 1) ; i++ )
  {
    #pragma HLS unroll
    c<<= 2;   
    #define UPPERBITS(value) (value>>30)
    c += UPPERBITS(value);
    #undef UPPERBITS
    value <<= 2;
    a <<= 1;
    b = (a<<1) | 1;
    if ( c >= b )
    {
      c -= b;
      a++;
    }
  }
  return a;
}

int max
(
  int a,
   int b
)
{
 if ( a > b )
   return a;
 else 
   return b;
}

int min
(
  int a,
  int b
)
{
  if ( a < b )
    return a;
  else 
    return b;
}

