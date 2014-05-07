/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2014, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */
// ------------------------------------------------------
//  Refer to the description in the wiki:
//  http://www.mrpt.org/Kalman_Filters
// ------------------------------------------------------

#include <mrpt/base.h>
#include <mrpt/gui.h>
#include <mrpt/obs.h>

#include <mrpt/bayes/CKalmanFilterCapable.h>

using namespace mrpt;
using namespace mrpt::bayes;
using namespace mrpt::gui;
using namespace mrpt::math;
using namespace mrpt::slam;
using namespace mrpt::utils;
using namespace mrpt::random;
using namespace std;

#define BEARING_SENSOR_NOISE_STD 	DEG2RAD(15.0f)
#define RANGE_SENSOR_NOISE_STD 		0.3f
#define DELTA_TIME                  	0.1f

#define VEHICLE_INITIAL_X			4.0f
#define VEHICLE_INITIAL_Y			4.0f
#define VEHICLE_INITIAL_V           1.0f
#define VEHICLE_INITIAL_W           DEG2RAD(20.0f)

#define TRANSITION_MODEL_STD_XY 	0.03f
#define TRANSITION_MODEL_STD_VXY 	0.20f

// ------------------------------------------------------
//				Configuration
// ------------------------------------------------------
CConfigFile		*iniFile = NULL;
std::string		iniFileName;

// Uncomment to save text files with grount truth vs. estimated states
//#define SAVE_GT_LOGS

// ------------------------------------------------------
//		Implementation of the system models as a EKF
// ------------------------------------------------------

/* ++ Anjan
 			VEH_SIZE: The dimension of the "vehicle state": either the full state vector or the "vehicle" part if applicable.
		 *	- OBS_SIZE: The dimension of each observation (eg, 2 for pixel coordinates, 3 for 3D coordinates,etc).
		 *	- FEAT_SIZE: The dimension of the features in the system state (the "map"), or 0 if not applicable (the default if not implemented).
		 *	- ACT_SIZE: The dimension of each "action" u_k (or 0 if not applicable).
		 *	- KFTYPE: The numeric type of the matrices (default: double)
*/
class CRangeBearing :
	public mrpt::bayes::CKalmanFilterCapable<4 /* x y vx vy*/, 2 /* range yaw */, 0               , 1 /* Atime */>
						 // <size_t VEH_SIZE,  size_t OBS_SIZE,  size_t FEAT_SIZE, size_t ACT_SIZE, size typename kftype = double>
{
public:
	CRangeBearing( );
	virtual ~CRangeBearing();

	void  doProcess( double DeltaTime, double observationRange, double observationBearing );

	void getState( KFVector &xkk, KFMatrix &pkk)
	{
		xkk = m_xkk;
		pkk = m_pkk;
	}

 protected:

	float m_obsBearing,m_obsRange;
	float m_deltaTime;
	/* State vectors */

	/** @name Virtual methods for Kalman Filter implementation
		@{
	 */

	/** Must return the action vector u.
	  * \param out_u The action vector which will be passed to OnTransitionModel
	  */
	void OnGetAction( KFArray_ACT &out_u ) const;

	/** Implements the transition model \f$ \hat{x}_{k|k-1} = f( \hat{x}_{k-1|k-1}, u_k ) \f$
	  * \param in_u The vector returned by OnGetAction.
	  * \param inout_x At input has \f$ \hat{x}_{k-1|k-1} \f$, at output must have \f$ \hat{x}_{k|k-1} \f$.
	  * \param out_skip Set this to true if for some reason you want to skip the prediction step (to do not modify either the vector or the covariance). Default:false
	  */
	void OnTransitionModel(
		const KFArray_ACT &in_u,
		KFArray_VEH       &inout_x,
		bool &out_skipPrediction
		 ) const;

	/** Implements the transition Jacobian \f$ \frac{\partial f}{\partial x} \f$
	  * \param out_F Must return the Jacobian.
	  *  The returned matrix must be \f$N \times N\f$ with N being either the size of the whole state vector or get_vehicle_size().
	  */
	void OnTransitionJacobian(KFMatrix_VxV  &out_F ) const;

	/** Implements the transition noise covariance \f$ Q_k \f$
	  * \param out_Q Must return the covariance matrix.
	  *  The returned matrix must be of the same size than the jacobian from OnTransitionJacobian
	  */
	void OnTransitionNoise(KFMatrix_VxV &out_Q ) const;

	/** Return the observation NOISE covariance matrix, that is, the model of the Gaussian additive noise of the sensor.
	  * \param out_R The noise covariance matrix. It might be non diagonal, but it'll usually be.
	  * \note Upon call, it can be assumed that the previous contents of out_R are all zeros.
	  */
	void OnGetObservationNoise(KFMatrix_OxO &out_R) const;

	/** This is called between the KF prediction step and the update step, and the application must return the observations and, when applicable, the data association between these observations and the current map.
	  *
	  * \param out_z N vectors, each for one "observation" of length OBS_SIZE, N being the number of "observations": how many observed landmarks for a map, or just one if not applicable.
	  * \param out_data_association An empty vector or, where applicable, a vector where the i'th element corresponds to the position of the observation in the i'th row of out_z within the system state vector (in the range [0,getNumberOfLandmarksInTheMap()-1]), or -1 if it is a new map element and we want to insert it at the end of this KF iteration.
	  * \param in_all_predictions A vector with the prediction of ALL the landmarks in the map. Note that, in contrast, in_S only comprises a subset of all the landmarks.
	  * \param in_S The full covariance matrix of the observation predictions (i.e. the "innovation covariance matrix"). This is a M·O x M·O matrix with M=length of "in_lm_indices_in_S".
	  * \param in_lm_indices_in_S The indices of the map landmarks (range [0,getNumberOfLandmarksInTheMap()-1]) that can be found in the matrix in_S.
	  *
	  *  This method will be called just once for each complete KF iteration.
	  * \note It is assumed that the observations are independent, i.e. there are NO cross-covariances between them.
	  */
	void OnGetObservationsAndDataAssociation(
		vector_KFArray_OBS			&out_z,
		mrpt::vector_int            &out_data_association,
		const vector_KFArray_OBS	&in_all_predictions,
		const KFMatrix              &in_S,
		const vector_size_t         &in_lm_indices_in_S,
		const KFMatrix_OxO          &in_R
		);

		/** Implements the observation prediction \f$ h_i(x) \f$.
		  * \param idx_landmark_to_predict The indices of the landmarks in the map whose predictions are expected as output. For non SLAM-like problems, this input value is undefined and the application should just generate one observation for the given problem.
		  * \param out_predictions The predicted observations.
		  */
		void OnObservationModel(
			const vector_size_t &idx_landmarks_to_predict,
			vector_KFArray_OBS  &out_predictions
			) const;

		/** Implements the observation Jacobians \f$ \frac{\partial h_i}{\partial x} \f$ and (when applicable) \f$ \frac{\partial h_i}{\partial y_i} \f$.
		  * \param idx_landmark_to_predict The index of the landmark in the map whose prediction is expected as output. For non SLAM-like problems, this will be zero and the expected output is for the whole state vector.
		  * \param Hx  The output Jacobian \f$ \frac{\partial h_i}{\partial x} \f$.
		  * \param Hy  The output Jacobian \f$ \frac{\partial h_i}{\partial y_i} \f$.
		  */
		void OnObservationJacobians(
			const size_t &idx_landmark_to_predict,
			KFMatrix_OxV &Hx,
			KFMatrix_OxF &Hy
			) const;

		/** Computes A=A-B, which may need to be re-implemented depending on the topology of the individual scalar components (eg, angles).
		  */
		void OnSubstractObservationVectors(KFArray_OBS &A, const KFArray_OBS &B) const;

	/** @}
	 */
};



// ------------------------------------------------------
//				GreenhouseLocalization
// ------------------------------------------------------
void GreenhouseLocalization()
{
	size_t	rawlogEntry, rawlogEntries;
	int firstObservation=0;
	TTimeStamp  previousTime;
	double delta_time;
	struct IMUStateVector{
	double m_yaw,m_roll,m_pitch,m_x_acc,m_y_acc,m_z_acc; /*Measured data */
	}imuSV;
	 
	// Load configuration!
	// ------------------------------------------

	std::string RAWLOG_FILE = iniFile->read_string("greenhouse-localization","rawlog_file","NOT FOUND!");
	ASSERT_FILE_EXISTS_(RAWLOG_FILE)

	// --------------------------
	// Load the rawlog:
	// --------------------------
	printf("Loading the rawlog file...");
	CRawlog			rawlog;
	rawlog.loadFromRawLogFile(RAWLOG_FILE);
	
	CFileGZInputStream      rawlogFile( RAWLOG_FILE );
	rawlogEntries = rawlog.size();
	std::cout<<"rawlog Entries: "<<rawlogEntries<<"\n";
	printf("OK\n");
	//mrpt::system::os::getch();
	CMatrixDouble 	data_matrix(rawlogEntries,16);

	rawlogEntry = 0;
	CActionCollectionPtr action;
	CSensoryFramePtr     SF;
	CObservationPtr     observations;
	CPose2D	pdfEstimation;
	CMatrixDouble33 rotationMatrix, rotationMatrixPrevious ,identityMatrix, res, skewMatrix, num,denom, skewMatrixDenom, temp;
	 CMatrixDouble m_acc(3,1), acc_updated(3,1), velocityPrevious(3,1), velocity(3,1), rangePrevious(3,1), range(3,1);
	//CVectorDouble vect_temp(3);
/*	m_acc(0,0) =0;
	m_acc(1,0) =1; 
	m_acc(2,0) =2; */
	rotationMatrix.unit();
	rotationMatrixPrevious.unit();
	identityMatrix.unit();
	identityMatrix.multiply_Ab(2,identityMatrix);
 //	printf("I(01) %f (02) %f \n",identityMatrix(0,1),identityMatrix(0,2));
//	printf("I(00) %f (11) %f  \n",identityMatrix(0,0),identityMatrix(1,1)); 
	/* for(int i=0; i<=2;i++)
		for(int j=0;j<=2;j++)
		{
			rotationMatrixPrevious(i,j) = 1;
			rotationMatrix(i,j) =1;
			
		} */

	while(rawlogEntry<rawlogEntries-1)
	{
		//cout << endl << "RAWLOG_ENTRY: " << rawlogEntry << endl << endl;

		if (! CRawlog::getActionObservationPairOrObservation( rawlogFile, action,SF, observations, rawlogEntry) )
                        break; // file EOF
 		if (IS_CLASS(observations,CObservationIMU))
		{
			CObservationIMUPtr imu = CObservationIMUPtr(observations);
			if(!firstObservation)
			{
				delta_time = 0;	
				firstObservation = 1;
			}
			else
				delta_time =  timeDifference(previousTime,imu->timestamp);
			previousTime = imu->timestamp;
		/*	cout << format("   IMU angles (degrees): (yaw,pitch,roll)=(%.06f, %.06f, %.06f )",
			RAD2DEG( imu->rawMeasurements[IMU_YAW] ),
			RAD2DEG( imu->rawMeasurements[IMU_PITCH] ),
			RAD2DEG( imu->rawMeasurements[IMU_ROLL] ) ) << endl; 
			cout<< "delta time : "<<delta_time;
		
			cout<< format("IMU accleration (x y z) = ( %f %f %f) ",
				imu->rawMeasurements[IMU_X_ACC],
				imu->rawMeasurements[IMU_Y_ACC], 
				imu->rawMeasurements[IMU_Z_ACC]) << endl;
		
			cout<< format("IMU angular velocity (YAW PITCH ROLL) = ( %f %f %f) ",
				imu->rawMeasurements[IMU_YAW_VEL],
				imu->rawMeasurements[IMU_PITCH_VEL], 
				imu->rawMeasurements[IMU_ROLL_VEL]) << endl;
		*/
		imuSV.m_yaw = imu->rawMeasurements[IMU_YAW_VEL];
		imuSV.m_roll = imu->rawMeasurements[IMU_ROLL_VEL];
		imuSV.m_pitch = imu->rawMeasurements[IMU_PITCH_VEL];

	/*	cout<< format("IMU accleration (x y z) = ( %f %f %f) ",
				imu->rawMeasurements[IMU_X_ACC],
				imu->rawMeasurements[IMU_Y_ACC], 
				imu->rawMeasurements[IMU_Z_ACC]) << endl; */

		m_acc(0,0) =imu->rawMeasurements[IMU_X_ACC];
		m_acc(1,0) =imu->rawMeasurements[IMU_Y_ACC];
		m_acc(2,0) =imu->rawMeasurements[IMU_Z_ACC];
		}
	
	skewMatrix(0,1) = -imuSV.m_yaw	*	delta_time;
	skewMatrix(0,2) =  imuSV.m_pitch*	delta_time;
	skewMatrix(1,0) =  imuSV.m_yaw	*	delta_time;
	skewMatrix(1,2) = -imuSV.m_roll	*	delta_time;
	skewMatrix(2,0) = -imuSV.m_pitch*	delta_time;
	skewMatrix(2,1) =  imuSV.m_roll	*	delta_time;

	/*res.multiply_AB(rotationMatrixPrevious,identityMatrix);
		identityMatrix = identityMatrix*6; */

	num = identityMatrix	+	skewMatrix;
	denom = identityMatrix	-	skewMatrix;

/*printf("num \n");
	printf(" %f  %f  %f\n",num(0,0),num(0,1),num(0,2));
	printf(" %f  %f  %f\n",num(1,0),num(1,1),num(1,2));
	printf(" %f  %f  %f\n",num(2,0),num(2,1),num(2,2));
printf("denom \n");
	printf(" %f  %f  %f\n",denom(0,0),denom(0,1),denom(0,2));
	printf(" %f  %f  %f\n",denom(1,0),denom(1,1),denom(1,2));
	printf(" %f  %f  %f\n",denom(2,0),denom(2,1),denom(2,2)); */

	//printf("yaw roll pitch %f  %f  %f \n",measuredYaw,measuredRoll,measuredPitch );
	//printf(" numm yaw roll pitch %f  %f  %f \n",measuredYaw,measuredRoll,measuredPitch );
	//res = num/deno;
//	deno.setSize(4,4);

	temp = denom.inv();

/*printf("denom inv \n");
	printf(" %f  %f  %f\n",temp(0,0),temp(0,1),temp(0,2));
	printf(" %f  %f  %f\n",temp(1,0),temp(1,1),temp(1,2));
	printf(" %f  %f  %f\n",temp(2,0),temp(2,1),temp(2,2)); */

	res.multiply_AB(num,temp);

/* printf("res \n");
	printf(" %f  %f  %f\n",res(0,0),res(0,1),res(0,2));
	printf(" %f  %f  %f\n",res(1,0),res(1,1),res(1,2));
	printf(" %f  %f  %f\n",res(2,0),res(2,1),res(2,2)); */
	rotationMatrix.multiply_AB(rotationMatrixPrevious, res);
/*printf("rotation \n");
	printf(" %f  %f  %f\n",rotationMatrix(0,0),rotationMatrix(0,1),rotationMatrix(0,2));
	printf(" %f  %f  %f\n",rotationMatrix(1,0),rotationMatrix(1,1),rotationMatrix(1,2));
	printf(" %f  %f  %f\n",rotationMatrix(2,0),rotationMatrix(2,1),rotationMatrix(2,2));*/

	rotationMatrixPrevious = rotationMatrix;
//	Eigen::Vector3d acc_up = rotationMatrix * m_acc;
	m_acc(2,0) -=9.8; 
	acc_updated.multiply(rotationMatrix, m_acc);	
	//printf("acc(x) %f (y) %f  (z) %f\n",acc_updated(0,0),acc_updated(1,0),acc_updated(2,0));
	
	acc_updated(0,0) *=delta_time;
 	acc_updated(1,0) *=delta_time;
	acc_updated(2,0) *=delta_time; 
//	printf("acc*deltatime (x) %f (y) %f  (z) %f\n",acc_updated(0,0),acc_updated(1,0),acc_updated(2,0));
	velocity = velocityPrevious + acc_updated;
	velocityPrevious = velocity;
//	printf("velocity (x) %f (y) %f  (z) %f\n\n",velocity(0,0),velocity(1,0),velocity(2,0));
	}

printf("anjan ok \n");
	randomGenerator.randomize();

	CDisplayWindowPlots		winEKF("Tracking - Extended Kalman Filter",450,400);

	winEKF.setPos(10,10);

	winEKF.axis(-2,20,-10,10); winEKF.axis_equal();


	// Create EKF
	// ----------------------
	CRangeBearing 	EKF;
	EKF.KF_options.method = kfEKFNaive;

	EKF.KF_options.verbose = true;
	EKF.KF_options.enable_profiler = true;

#ifdef SAVE_GT_LOGS
	CFileOutputStream  fo_log_ekf("log_GT_vs_EKF.txt");
	fo_log_ekf.printf("%%%% GT_X  GT_Y  EKF_MEAN_X  EKF_MEAN_Y   EKF_STD_X   EKF_STD_Y\n");
#endif

	// Init. simulation:
	// -------------------------
	float x=VEHICLE_INITIAL_X,y=VEHICLE_INITIAL_Y,phi=DEG2RAD(-180),v=VEHICLE_INITIAL_V,w=VEHICLE_INITIAL_W;
	float  t=0;

	while (winEKF.isOpen() && 
 !mrpt::system::os::kbhit() )
	{
		// Update vehicle:
		x+=v*DELTA_TIME*(cos(phi)-sin(phi));
		y+=v*DELTA_TIME*(sin(phi)+cos(phi));
		phi+=w*DELTA_TIME;

		v+=1.0f*DELTA_TIME*cos(t);
		w-=0.1f*DELTA_TIME*sin(t);


		// Simulate noisy observation:
		float realBearing = atan2( y,x );
		float obsBearing = realBearing  + BEARING_SENSOR_NOISE_STD * randomGenerator.drawGaussian1D_normalized();
		printf("Real/Simulated bearing: %.03f / %.03f deg\n", RAD2DEG(realBearing), RAD2DEG(obsBearing) );

		float realRange = sqrt(square(x)+square(y));
		float obsRange = max(0.0, realRange  + RANGE_SENSOR_NOISE_STD * randomGenerator.drawGaussian1D_normalized() );
		printf("Real/Simulated range: %.03f / %.03f \n", realRange, obsRange );

		// Process with EKF:
		EKF.doProcess(DELTA_TIME,obsRange, obsBearing);

		// Show EKF state:
		CRangeBearing::KFVector EKF_xkk;
		CRangeBearing::KFMatrix EKF_pkk;

		EKF.getState( EKF_xkk, EKF_pkk );

		printf("Real: x:%.03f  y=%.03f heading=%.03f v=%.03f w=%.03f\n",x,y,phi,v,w);
		cout << "EKF: " << EKF_xkk << endl;

		// Draw EKF state:
		CRangeBearing::KFMatrix   COVXY(2,2);
		COVXY(0,0) = EKF_pkk(0,0);
		COVXY(1,1) = EKF_pkk(1,1);
		COVXY(0,1) = COVXY(1,0) = EKF_pkk(0,1);

		winEKF.plotEllipse( EKF_xkk[0], EKF_xkk[1], COVXY, 3, "b-2", "ellipse_EKF" );

		// Save GT vs EKF state:
#ifdef SAVE_GT_LOGS
		// %% GT_X  GT_Y  EKF_MEAN_X  EKF_MEAN_Y   EKF_STD_X   EKF_STD_Y:
		fo_log_ekf.printf("%f %f %f %f %f %f\n",
			x,y, // Real (GT)
			EKF_xkk[0], EKF_xkk[1],
			std::sqrt(EKF_pkk(0,0)), std::sqrt(EKF_pkk(1,1))
			);
#endif

		// Draw the velocity vector:
		vector_float vx(2),vy(2);
		vx[0] = EKF_xkk[0];  vx[1] = vx[0] + EKF_xkk[2] * 1;
		vy[0] = EKF_xkk[1];  vy[1] = vy[0] + EKF_xkk[3] * 1;
		winEKF.plot( vx,vy, "g-4", "velocityEKF" );
		// Draw GT:
		winEKF.plot( vector_float(1,x), vector_float(1,y),"k.8","plot_GT");
		// Draw noisy observations:
		vector_float  obs_x(2),obs_y(2);
		obs_x[0] = obs_y[0] = 0;
		obs_x[1] = obsRange * cos( obsBearing );
		obs_y[1] = obsRange * sin( obsBearing );
		winEKF.plot(obs_x,obs_y,"r", "plot_obs_ray");
		// Delay:
		mrpt::system::sleep((int)(DELTA_TIME*1000));
		t+=DELTA_TIME;
	}
}


// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------
int main(int argc, char **argv)
{
	try
	{
		printf(" Greenhouse Localization - Version 0.1 \n");
		printf(" MRPT C++ Library: %s \n", MRPT_getVersion().c_str());
		printf("-------------------------------------------------------------------\n");

		if (argc<2)
		{
			printf("Usage: %s <config.ini>\n\n",argv[0]);
			//pause();
			return -1;
		}

		iniFileName = argv[1];

		iniFile = new CConfigFile(iniFileName);
		printf(" Greenhouse Localization - Version 0.1 \n");
		GreenhouseLocalization();
		delete iniFile;
		return 0;

	} catch (std::exception &e)
	{
		std::cout << "MRPT exception caught: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		printf("Untyped exception!!");
		return -1;
	}
}




CRangeBearing::CRangeBearing()
{
	//KF_options.method = kfEKFNaive;
	KF_options.method = kfEKFAlaDavison;

	// INIT KF STATE
	m_xkk.resize(4,0);	// State: (x,y,heading,v,w)
	m_xkk[0]= VEHICLE_INITIAL_X;
	m_xkk[1]= VEHICLE_INITIAL_Y;
	m_xkk[2]=-VEHICLE_INITIAL_V;
	m_xkk[3]=0;

	// Initial cov:  Large uncertainty
	m_pkk.setSize(4,4);
	m_pkk.unit();
	m_pkk(0,0)=
	m_pkk(1,1)= square( 5.0f );
	m_pkk(2,2)=
	m_pkk(3,3)= square( 1.0f );
}

CRangeBearing::~CRangeBearing()
{

}


void  CRangeBearing::doProcess( double DeltaTime, double observationRange, double observationBearing )
{
	m_deltaTime = (float)DeltaTime;
	m_obsBearing = (float)observationBearing;
	m_obsRange = (float) observationRange;

	runOneKalmanIteration();
}


/** Must return the action vector u.
  * \param out_u The action vector which will be passed to OnTransitionModel
  */
void CRangeBearing::OnGetAction( KFArray_ACT &u ) const
{
	u[0] = m_deltaTime;
}

/** Implements the transition model \f$ \hat{x}_{k|k-1} = f( \hat{x}_{k-1|k-1}, u_k ) \f$
  * \param in_u The vector returned by OnGetAction.
  * \param inout_x At input has \f$ \hat{x}_{k-1|k-1} \f$, at output must have \f$ \hat{x}_{k|k-1} \f$.
  * \param out_skip Set this to true if for some reason you want to skip the prediction step (to do not modify either the vector or the covariance). Default:false
  */
void CRangeBearing::OnTransitionModel(
	const KFArray_ACT &in_u,
	KFArray_VEH       &inout_x,
	bool &out_skipPrediction
	) const
{
	//printf("Anjan in OnTransitionModel");
	// in_u[0] : Delta time
	// in_out_x: [0]:x  [1]:y  [2]:vx  [3]: vy
	inout_x[0] += in_u[0] * inout_x[2];
	inout_x[1] += in_u[0] * inout_x[3];

}

/** Implements the transition Jacobian \f$ \frac{\partial f}{\partial x} \f$
  * \param out_F Must return the Jacobian.
  *  The returned matrix must be \f$N \times N\f$ with N being either the size of the whole state vector or get_vehicle_size().
  */
void CRangeBearing::OnTransitionJacobian(KFMatrix_VxV  &F) const
{
	//printf("Anjan in OnTransitionJacobian");
	F.unit();

	F(0,2) = m_deltaTime;
	F(1,3) = m_deltaTime;
}

/** Implements the transition noise covariance \f$ Q_k \f$
  * \param out_Q Must return the covariance matrix.
  *  The returned matrix must be of the same size than the jacobian from OnTransitionJacobian
  */
void CRangeBearing::OnTransitionNoise(KFMatrix_VxV &Q) const
{
	//printf("Anjan in OnTransitionNoise");
	Q(0,0) =
	Q(1,1) = square( TRANSITION_MODEL_STD_XY );
	Q(2,2) =
	Q(3,3) = square( TRANSITION_MODEL_STD_VXY );
}

/** Return the observation NOISE covariance matrix, that is, the model of the Gaussian additive noise of the sensor.
* \param out_R The noise covariance matrix. It might be non diagonal, but it'll usually be.
* \note Upon call, it can be assumed that the previous contents of out_R are all zeros.
*/
void CRangeBearing::OnGetObservationNoise(KFMatrix_OxO &R) const
{
	//printf("Anjan in OnGetObservationNoise");
	R(0,0) = square( BEARING_SENSOR_NOISE_STD );
	R(1,1) = square( RANGE_SENSOR_NOISE_STD );
}

void CRangeBearing::OnGetObservationsAndDataAssociation(
	vector_KFArray_OBS			&out_z,
	mrpt::vector_int            &out_data_association,
	const vector_KFArray_OBS	&in_all_predictions,
	const KFMatrix              &in_S,
	const vector_size_t         &in_lm_indices_in_S,
	const KFMatrix_OxO          &in_R
	)
{
	//printf("Anjan in OnGetObservationsAndDataAssociation");
	out_z.resize(1);
	out_z[0][0] = m_obsBearing;
	out_z[0][1] = m_obsRange;

	out_data_association.clear(); // Not used
}


/** Implements the observation prediction \f$ h_i(x) \f$.
  * \param idx_landmark_to_predict The indices of the landmarks in the map whose predictions are expected as output. For non SLAM-like problems, this input value is undefined and the application should just generate one observation for the given problem.
  * \param out_predictions The predicted observations.
  */
void CRangeBearing::OnObservationModel(
	const vector_size_t       &idx_landmarks_to_predict,
	vector_KFArray_OBS	&out_predictions
	) const
{
	//printf("Anjan in OnObservationModel");
	// predicted bearing:
	kftype x = m_xkk[0];
	kftype y = m_xkk[1];

	kftype h_bear = atan2(y,x);
	kftype h_range = sqrt(square(x)+square(y));

	// idx_landmarks_to_predict is ignored in NON-SLAM problems
	out_predictions.resize(1);
	out_predictions[0][0] = h_bear;
	out_predictions[0][1] = h_range;
}

/** Implements the observation Jacobians \f$ \frac{\partial h_i}{\partial x} \f$ and (when applicable) \f$ \frac{\partial h_i}{\partial y_i} \f$.
  * \param idx_landmark_to_predict The index of the landmark in the map whose prediction is expected as output. For non SLAM-like problems, this will be zero and the expected output is for the whole state vector.
  * \param Hx  The output Jacobian \f$ \frac{\partial h_i}{\partial x} \f$.
  * \param Hy  The output Jacobian \f$ \frac{\partial h_i}{\partial y_i} \f$.
  */
void CRangeBearing::OnObservationJacobians(
	const size_t &idx_landmark_to_predict,
	KFMatrix_OxV &Hx,
	KFMatrix_OxF &Hy
	) const
{
	//printf("Anjan in OnObservationJacobians");
	// predicted bearing:
	kftype x = m_xkk[0];
	kftype y = m_xkk[1];

	Hx.zeros();
	Hx(0,0) = -y/(square(x)+square(y));
	Hx(0,1) = 1/(x*(1+square(y/x)));

	Hx(1,0) = x/sqrt(square(x)+square(y));
	Hx(1,1) = y/sqrt(square(x)+square(y));

	// Hy: Not used
}

/** Computes A=A-B, which may need to be re-implemented depending on the topology of the individual scalar components (eg, angles).
  */
void CRangeBearing::OnSubstractObservationVectors(KFArray_OBS &A, const KFArray_OBS &B) const
{
	A -= B;
	math::wrapToPiInPlace(A[0]); // The angular component
}

