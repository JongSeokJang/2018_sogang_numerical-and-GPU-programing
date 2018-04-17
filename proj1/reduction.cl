__kernel void Reduction_scalar(__global float* data, __local float* partial_sums, __global float *output ){

	int lid = get_local_id(0);
	int group_size = get_local_size(0);
	partial_sums[lid] = data[get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for( int i = group_size/2; i>0; i >>= 1){
		if(lid < i){
			partial_sums[lid] += partial_sums[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}

	if(lid == 0){
		output[get_group_id(0)] = partial_sums[0];	
	}
}



__kernel void Reduction_global(__global float* data,  __global float *output ){

	int gid = get_global_id(0);
	int group_size = get_local_size(0);
	int lid = get_local_id(0);

	for( int i = group_size/2; i>0; i >>= 1){
		if(lid < i){ 
			data[gid] += data[gid + i];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	if(lid == 0){
		output[get_group_id(0)] = data[gid];	
	}
	
}



__kernel void Reduction_scalar_2D(__global float* data, __local float* partial_sums, __global float *output ){


	int group_size = get_local_size(0) * get_local_size(1);
	int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);

	partial_sums[ lid ] = data[ get_global_size(0) * get_global_id(1) + get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for( int i = group_size/2; i>0; i >>= 1){
		if(lid < i){
			partial_sums[lid] += partial_sums[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}

	if(lid == 0){
		output[get_group_id(1) * get_num_groups(0) + get_group_id(0)] = partial_sums[0];	
	}
}



__kernel void Reduction_global_2D(__global float* data, __global float *output ){

	
	int group_size = get_local_size(0) * get_local_size(1);

	int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
	int gid	= get_global_id(1) * get_global_size(0) + get_global_id(0);
	
	for( int i = group_size/2; i>0; i >>= 1){
		if(lid < i){
			data[gid] += data[gid + (i/get_local_size(0))*get_global_size(0) + i % get_local_size(0)];
			//data[gid] += data[ (get_global_id(1) + (i / get_local_size(0))) * get_global_size(0) + (get_global_id(0) + i % get_local_size(0) )];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

	}

	if(lid == 0){
		output[get_group_id(1) * get_num_groups(0) + get_group_id(0)] = data[gid];	
	}
}
