/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdlib.h>
#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xtoptop.h"
#include "xparameters.h"
#include "xtime_l.h"
#include "toptop.h"
#include "xuartps.h"
#include "math.h"

int main()
{
    init_platform();
    Xil_DCacheDisable();

	XTime t1,t2;
	print("Lifetime computing (fixed-point)\n");

	XToptop Xtoptop;
    int status;
    float tau_inten1,tau_amp1,tau_inten2,tau_amp2,tau_inten3,tau_amp3,tau_inten4,tau_amp4;

    status=XToptop_Initialize(&Xtoptop,XPAR_TOPTOP_0_DEVICE_ID);
    printf("Xtoptop IP device initialized status=%d\n",status);

/********************************UART configuration**************************************/
	XUartPs myUart;
	XUartPs_Config *myUartConfig;
	myUartConfig = XUartPs_LookupConfig(XPAR_PS7_UART_0_DEVICE_ID);
	status = XUartPs_CfgInitialize(&myUart,myUartConfig,XPAR_PS7_UART_0_BASEADDR);
	if (status != XST_SUCCESS)
		printf("--------UART initialized failed---------\n");
	/*  UART self test  */
	status = XUartPs_SelfTest(&myUart);
	if (status != XST_SUCCESS)
		printf("--------UART self test failed-----------\n");

	u32 BAUDRATE=921600;
	XUartPsFormat uart_format =
	{
		BAUDRATE,   // 115200
		XUARTPS_FORMAT_8_BITS,
		XUARTPS_FORMAT_NO_PARITY,
		XUARTPS_FORMAT_1_STOP_BIT,
	};
	XUartPs_SetDataFormat(&myUart, &uart_format);    //Configure UART format
	printf("------------UART config done------------ \n");
	XUartPs_SetOperMode(&myUart, XUARTPS_OPER_MODE_NORMAL);//Operation mode
	printf("------------UART mode config done------------ \n");

/******************************PL accelerator configuration*********************************/
	int l=0;
	u8 hist_data[4*2*bin_number];//256(time bin)*4(cores)*2(q and r)
	int y_comb1[bin_number],y_comb2[bin_number],y_comb3[bin_number],y_comb4[bin_number];
    int Lifetimes_out1[2],Lifetimes_out2[2],Lifetimes_out3[2],Lifetimes_out4[2];
    u8 lifetime_buffer[4*2*2];//8 lifetimes from 4 cores, times 2 (q and r, decode factor to transfer to PC via UART)
    float sum_cost;
	while(l<pixel_number){
		u32 receivedBytes=0;
		u32 totalreceivedBytes=0;
		u32 sentBytes=0;
		u32 totalsentBytes=0;

		XTime_GetTime(&t1);//Start
		while(totalreceivedBytes <  4*2*bin_number*sizeof(u8))//*2, due to the y_q and y_r, wait here if no data received.
		{
			receivedBytes = XUartPs_Recv(&myUart, (u8*)&hist_data[totalreceivedBytes], sizeof(u8));
			totalreceivedBytes+=receivedBytes;
		}
		for (int i=0;i<bin_number;i++)
		{
			y_comb1[i]=hist_data[i]*256+hist_data[i+256];
			y_comb2[i]=hist_data[i+512]*256+hist_data[i+768];
			y_comb3[i]=hist_data[i+1024]*256+hist_data[i+1280];
			y_comb4[i]=hist_data[i+1536]*256+hist_data[i+1792];
		}

	    XTime_GetTime(&t2);
		float tmp =(t2-t1);
		sum_cost= (float)tmp/COUNTS_PER_SECOND;

	    toptop_FixP(&Xtoptop,y_comb1,y_comb2,y_comb3,y_comb4,
	    		Lifetimes_out1,Lifetimes_out2,Lifetimes_out3,Lifetimes_out4);

	    lifetime_buffer[0]=floor(Lifetimes_out1[0]/256);//Core #1
	    lifetime_buffer[1]=Lifetimes_out1[0]%256;
	    lifetime_buffer[2]=floor(Lifetimes_out1[1]/256);
	    lifetime_buffer[3]=Lifetimes_out1[1]%256;

	    lifetime_buffer[4]=floor(Lifetimes_out2[0]/256);//Core #2
	    lifetime_buffer[5]=Lifetimes_out2[0]%256;
	    lifetime_buffer[6]=floor(Lifetimes_out2[1]/256);
	    lifetime_buffer[7]=Lifetimes_out2[1]%256;

	    lifetime_buffer[8]=floor(Lifetimes_out3[0]/256);//Core #3
	    lifetime_buffer[9]=Lifetimes_out3[0]%256;
	    lifetime_buffer[10]=floor(Lifetimes_out3[1]/256);
	    lifetime_buffer[11]=Lifetimes_out3[1]%256;

	    lifetime_buffer[12]=floor(Lifetimes_out4[0]/256);//Core #4
	    lifetime_buffer[13]=Lifetimes_out4[0]%256;
	    lifetime_buffer[14]=floor(Lifetimes_out4[1]/256);
	    lifetime_buffer[15]=Lifetimes_out4[1]%256;

		while(totalsentBytes <  2*PARA_NO*2*sizeof(u8))//To do
		{
			sentBytes = XUartPs_Send(&myUart, (u8*)&lifetime_buffer[totalsentBytes], sizeof(u8));
			totalsentBytes +=sentBytes;
		}

		l+=1;//Next pixel
	}
    cleanup_platform();
    return 0;
}
