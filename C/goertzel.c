void goertzel(float32_t *adcIn, float coef, float *goertzel_array, uint16_t adcLen, float32_t *maxVB){

	float s_prev = 0.0;
	float s_prev2 = 0.0;
	float s = 0.0;
	float power = 0;

	float32_t std;
	float32_t mean;

	arm_mean_f32(goertzel_array,20,&mean);

	arm_std_f32(goertzel_array,20,&std);


	 for (int i = 0; i < adcLen; i++) {
	        s = adcIn[i] + coef * s_prev - s_prev2;
	        s_prev2 = s_prev;
	        s_prev = s;
	    }

	power = s_prev2 * s_prev2 + s_prev * s_prev - coef * s_prev * s_prev2;

	goertzel_array[count_go_array] = power;

	power = (power / (mean+1*std))-1;

	count_go_array++;

	if(count_go_array == 32){
		count_go_array = 0;
	}

	*maxVB = power;
}
