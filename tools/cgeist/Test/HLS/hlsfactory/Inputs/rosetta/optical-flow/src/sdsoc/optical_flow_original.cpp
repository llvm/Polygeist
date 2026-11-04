/*===============================================================*/
/*                                                               */
/*                      optical_flow.cpp                         */
/*                                                               */
/*             Hardware function for optical flow                */
/*                                                               */
/*===============================================================*/

#include "optical_flow.h"
// use HLS video library
#include <hls_video.h>

// use HLS fixed point
#include "ap_fixed.h"

// define these constants so they can be used in pragma
const int max_width = MAX_WIDTH; 
const int default_depth = MAX_WIDTH;
// calculate gradient in x and y directions
void gradient_xy_calc(input_t frame[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH])
{
  // our own line buffer
  static pixel_t buf[5][MAX_WIDTH];
  #pragma HLS array_partition variable=buf complete dim=1

  // small buffer
  pixel_t smallbuf[5];
  #pragma HLS array_partition variable=smallbuf complete dim=0
  
  // window buffer
  hls::Window<5,5,input_t> window;

  const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};

  GRAD_XY_OUTER: for(int r=0; r<MAX_HEIGHT+2; r++)
  {
    GRAD_XY_INNER: for(int c=0; c<MAX_WIDTH+2; c++)
    {
      #pragma HLS pipeline II=1
      // read out values from current line buffer
      for (int i = 0; i < 4; i ++ )
        smallbuf[i] = buf[i+1][c];
      // the new value is either 0 or read from frame
      if (r<MAX_HEIGHT && c<MAX_WIDTH)
        smallbuf[4] = (pixel_t)(frame[r][c]);
      else if (c < MAX_WIDTH)
        smallbuf[4] = 0;
      // update line buffer
      if(r<MAX_HEIGHT && c<MAX_WIDTH)
      {
        for (int i = 0; i < 4; i ++ )
          buf[i][c] = smallbuf[i];
        buf[4][c] = smallbuf[4];
      }
      else if(c<MAX_WIDTH)
      {
        for (int i = 0; i < 4; i ++ )
          buf[i][c] = smallbuf[i];
        buf[4][c] = smallbuf[4];
      }

      // manage window buffer
      if(r<MAX_HEIGHT && c<MAX_WIDTH)
      {
        window.shift_pixels_left();
        
        for (int i = 0; i < 5; i ++ )
          window.insert_pixel(smallbuf[i],i,4);
      }
      else
      {
        window.shift_pixels_left();
        window.insert_pixel(0,0,4);
        window.insert_pixel(0,1,4);
        window.insert_pixel(0,2,4);
        window.insert_pixel(0,3,4);
        window.insert_pixel(0,4,4);
      }

      // compute gradient
      pixel_t x_grad = 0;
      pixel_t y_grad = 0;
      if(r>=4 && r<MAX_HEIGHT && c>=4 && c<MAX_WIDTH)
      {
        GRAD_XY_XYGRAD: for(int i=0; i<5; i++)
        {
          x_grad += window.getval(2,i)*GRAD_WEIGHTS[i];
          y_grad += window.getval(i,2)*GRAD_WEIGHTS[i];
        }
        gradient_x[r-2][c-2] = x_grad/12;
        gradient_y[r-2][c-2] = y_grad/12;
      }
      else if(r>=2 && c>=2)
      {
        gradient_x[r-2][c-2] = 0;
        gradient_y[r-2][c-2] = 0;
      }
    }
  }
}

// calculate gradient in the z direction
void gradient_z_calc(input_t frame1[MAX_HEIGHT][MAX_WIDTH], 
    input_t frame2[MAX_HEIGHT][MAX_WIDTH], 
    input_t frame3[MAX_HEIGHT][MAX_WIDTH], 
    input_t frame4[MAX_HEIGHT][MAX_WIDTH], 
    input_t frame5[MAX_HEIGHT][MAX_WIDTH], 
    pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH])
{
  const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};
  GRAD_Z_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    GRAD_Z_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      gradient_z[r][c] =((pixel_t)(frame1[r][c]*GRAD_WEIGHTS[0] 
                        + frame2[r][c]*GRAD_WEIGHTS[1]
                        + frame3[r][c]*GRAD_WEIGHTS[2]
                        + frame4[r][c]*GRAD_WEIGHTS[3]
                        + frame5[r][c]*GRAD_WEIGHTS[4]))/12;
    }
  }
}

// average the gradient in y direction
void gradient_weight_y(pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH],
    gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  hls::LineBuffer<7,MAX_WIDTH,gradient_t> buf;

  const pixel_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
  GRAD_WEIGHT_Y_OUTER: for(int r=0; r<MAX_HEIGHT+3; r++)
  {
    GRAD_WEIGHT_Y_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      #pragma HLS dependence variable=buf inter false

      if(r<MAX_HEIGHT)
      {
        buf.shift_pixels_up(c);
        gradient_t tmp;
        tmp.x = gradient_x[r][c];
        tmp.y = gradient_y[r][c];
        tmp.z = gradient_z[r][c];
        buf.insert_bottom_row(tmp,c);
      }
      else
      {
        buf.shift_pixels_up(c);
        gradient_t tmp;
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;
        buf.insert_bottom_row(tmp,c);
      }     

      gradient_t acc;
      acc.x = 0;
      acc.y = 0;
      acc.z = 0;
      if(r >= 6 && r<MAX_HEIGHT)
      { 
        GRAD_WEIGHT_Y_ACC: for(int i=0; i<7; i++)
        {
          acc.x += buf.getval(i,c).x*GRAD_FILTER[i];
          acc.y += buf.getval(i,c).y*GRAD_FILTER[i];
          acc.z += buf.getval(i,c).z*GRAD_FILTER[i];
        }
        filt_grad[r-3][c] = acc;
      }
      else if(r>=3)
      {
        filt_grad[r-3][c] = acc;
      }
    }
  }
}

// average gradient in the x direction
void gradient_weight_x(gradient_t y_filt[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  hls::Window<1,7,gradient_t> buf;
  const pixel_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
  GRAD_WEIGHT_X_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    GRAD_WEIGHT_X_INNER: for(int c=0; c<MAX_WIDTH+3; c++)
    {
      #pragma HLS pipeline II=1
      buf.shift_pixels_left();
      gradient_t tmp;
      if(c<MAX_WIDTH)
      {
        tmp = y_filt[r][c];
      }
      else
      {
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;  
      }
      buf.insert_pixel(tmp,0,6);

      gradient_t acc;
      acc.x = 0;
      acc.y = 0;
      acc.z = 0;
      if(c >= 6 && c<MAX_WIDTH) 
      {
        GRAD_WEIGHT_X_ACC: for(int i=0; i<7; i++)
        {
          acc.x += buf.getval(0,i).x*GRAD_FILTER[i];
          acc.y += buf.getval(0,i).y*GRAD_FILTER[i];
          acc.z += buf.getval(0,i).z*GRAD_FILTER[i];
        }
        filt_grad[r][c-3] = acc;
      }
      else if(c>=3)
      {
        filt_grad[r][c-3] = acc;
      }
    }
  }
}

// outer product 
void outer_product(gradient_t gradient[MAX_HEIGHT][MAX_WIDTH],
     outer_t outer_product[MAX_HEIGHT][MAX_WIDTH])
{
  OUTER_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    OUTER_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      gradient_t grad = gradient[r][c];
      outer_pixel_t x = (outer_pixel_t) grad.x;
      outer_pixel_t y = (outer_pixel_t) grad.y;
      outer_pixel_t z = (outer_pixel_t) grad.z;
      outer_t out;
      out.val[0] = (x*x);
      out.val[1] = (y*y);
      out.val[2] = (z*z);
      out.val[3] = (x*y);
      out.val[4] = (x*z);
      out.val[5] = (y*z);
      outer_product[r][c] = out;
    }
  }
}

// tensor weight
void tensor_weight_y(outer_t outer[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH])
{
  hls::LineBuffer<3,MAX_WIDTH,outer_t> buf;
  const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  TENSOR_WEIGHT_Y_OUTER: for(int r=0; r<MAX_HEIGHT+1; r++)
  {
    TENSOR_WEIGHT_Y_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      
      outer_t tmp;
      #pragma HLS data_pack variable=tmp
      #pragma HLS data_pack variable=buf.val[0]
      buf.shift_pixels_up(c);
      if(r<MAX_HEIGHT)
      {
        tmp = outer[r][c];
      }
      else
      {
        TENSOR_WEIGHT_Y_TMP_INIT: for(int i=0; i<6; i++)
          tmp.val[i] = 0;
      }   
      buf.insert_bottom_row(tmp,c);

      tensor_t acc;
      TENSOR_WEIGHT_Y_ACC_INIT: for(int k =0; k<6; k++)
        acc.val[k] = 0;
     
      if (r >= 2 && r < MAX_HEIGHT) 
      {
        TENSOR_WEIGHT_Y_TMP_OUTER: for(int i=0; i<3; i++)
        {
          tmp = buf.getval(i,c);
          pixel_t k = TENSOR_FILTER[i];
          TENSOR_WEIGHT_Y_TMP_INNER: for(int component=0; component<6; component++)
          {
            acc.val[component] += tmp.val[component]*k;
          }
        }
      }
      if(r >= 1)
      { 
        tensor_y[r-1][c] = acc;      
      }
    }
  }
}

void tensor_weight_x(tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor[MAX_HEIGHT][MAX_WIDTH])
{
  hls::Window<1,3,tensor_t> buf;
  const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  //const float TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  TENSOR_WEIGHT_X_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    TENSOR_WEIGHT_X_INNER: for(int c=0; c<MAX_WIDTH+1; c++)
    {
      #pragma HLS pipeline II=1
      buf.shift_pixels_left();
      tensor_t tmp;
      if(c<MAX_WIDTH)
      {
        tmp = tensor_y[r][c];
      }
      else
      {
        TENSOR_WEIGHT_X_TMP_INIT: for(int i=0; i<6; i++)
          tmp.val[i] = 0;
      }
      buf.insert_pixel(tmp,0,2);

      tensor_t acc;
      TENSOR_WEIGHT_X_ACC_INIT: for(int k =0; k<6; k++)
        acc.val[k] = 0;
      if (c >= 2 && c < MAX_WIDTH) 
      {
        TENSOR_WEIGHT_X_TMP_OUTER: for(int i=0; i<3; i++)
        {
          tmp = buf.getval(0,i);
          TENSOR_WEIGHT_X_TMP_INNER: for(int component=0; component<6; component++)
          {
            acc.val[component] += tmp.val[component]*TENSOR_FILTER[i];
          }
        }
      }
      if(c>=1)
      {
        tensor[r][c-1] = acc;
      }
    }
  }
}

// compute output flow
void flow_calc(tensor_t tensors[MAX_HEIGHT][MAX_WIDTH],
               velocity_t outputs[MAX_HEIGHT][MAX_WIDTH])
{
  static outer_pixel_t buf[2];
  FLOW_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    FLOW_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      tensor_t tmp_tensor = tensors[r][c];
      if(r>=2 && r<MAX_HEIGHT-2 && c>=2 && c<MAX_WIDTH-2)
      {
	      calc_pixel_t t1 = (calc_pixel_t) tmp_tensor.val[0];
	      calc_pixel_t t2 = (calc_pixel_t) tmp_tensor.val[1];
	      calc_pixel_t t3 = (calc_pixel_t) tmp_tensor.val[2];
	      calc_pixel_t t4 = (calc_pixel_t) tmp_tensor.val[3];
	      calc_pixel_t t5 = (calc_pixel_t) tmp_tensor.val[4];
	      calc_pixel_t t6 = (calc_pixel_t) tmp_tensor.val[5];

        calc_pixel_t denom = t1*t2-t4*t4;
	      calc_pixel_t numer0 = t6*t4-t5*t2;
	      calc_pixel_t numer1 = t5*t4-t6*t1;

	      if(denom != 0)
        {
          buf[0] = numer0 / denom;
          buf[1] = numer1 / denom;
	      } 
	      else
	      {
		      buf[0] = 0;
		      buf[1] = 0;
	      }
      }
      else
      {
        buf[0] = buf[1] = 0;
      }

      outputs[r][c].x = (vel_pixel_t)buf[0];
      outputs[r][c].y = (vel_pixel_t)buf[1];

    }
  }
}

// top-level kernel function
void optical_flow(frames_t   frames[MAX_HEIGHT][MAX_WIDTH],
                  velocity_t outputs[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS data_pack variable=outputs

  #pragma HLS DATAFLOW

  // FIFOs connecting the stages
  static pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=gradient_x depth=default_depth
  static pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=gradient_y depth=default_depth
  static pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=gradient_z depth=max_width*4
  static gradient_t y_filtered[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=y_filtered depth=default_depth
  static gradient_t filtered_gradient[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=filtered_gradient depth=default_depth
  static outer_t out_product[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=out_product depth=default_depth
  #pragma HLS data_pack variable=out_product
  static tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=tensor_y depth=default_depth
  #pragma HLS data_pack variable=tensor_y
  static tensor_t tensor[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=tensor depth=default_depth
  #pragma HLS data_pack variable=tensor

  // FIFOs for streaming in, just for clarity
  static input_t frame1_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame1_a depth=default_depth
  static input_t frame2_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame2_a depth=default_depth
  static input_t frame4_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame4_a depth=default_depth
  static input_t frame5_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame5_a depth=default_depth

  //Need to duplicate frame3 for the two calculations
  static input_t frame3_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame3_a depth=default_depth
  static input_t frame3_b[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame3_b depth=default_depth

  // stream in and organize the inputs
  static frames_t buf;
  FRAMES_CP_OUTER: for (int r=0; r<MAX_HEIGHT; r++) 
  {
    FRAMES_CP_INNER: for (int c=0; c<MAX_WIDTH; c++) 
    {
      #pragma HLS pipeline II=1

      // one wide read
      buf = frames[r][c];
      // assign values to the FIFOs
      frame1_a[r][c] = ((input_t)(buf(7 ,  0)) >> 8);
      frame2_a[r][c] = ((input_t)(buf(15,  8)) >> 8);
      frame3_a[r][c] = ((input_t)(buf(23, 16)) >> 8);
      frame3_b[r][c] = ((input_t)(buf(23, 16)) >> 8);
      frame4_a[r][c] = ((input_t)(buf(31, 24)) >> 8);
      frame5_a[r][c] = ((input_t)(buf(39, 32)) >> 8);
    }
  }
  //
  // compute
  gradient_xy_calc(frame3_a, gradient_x, gradient_y);
  gradient_z_calc(frame1_a, frame2_a, frame3_b, frame4_a, frame5_a, gradient_z);
  gradient_weight_y(gradient_x, gradient_y, gradient_z, y_filtered);
  gradient_weight_x(y_filtered, filtered_gradient);
  outer_product(filtered_gradient, out_product);
  tensor_weight_y(out_product, tensor_y);
  tensor_weight_x(tensor_y, tensor);
  flow_calc(tensor, outputs);

}
