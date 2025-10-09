# IoT device identification methods evalated:

## IoT Sentinel

**Paper**:
```
M. Miettinen, S. Marchal, I. Hafeez, N. Asokan, A. -R. Sadeghi and S. Tarkoma, "IoT SENTINEL: Automated Device-Type Identification for Security Enforcement in IoT," 2017 IEEE 37th International Conference on Distributed Computing Systems (ICDCS)
```

**Code Repository**: https://github.com/ChakshuGupta/IoT-Sentinel-device-identification

The implementation is forked from *nishadhperera/IoT-device-identification* and updated to meet the requirements for these evaluations.


## MUDgee

**Paper**: 
```
A. Hamza, D. Ranathunga, H. H. Gharakheili, T. A. Benson, M. Roughan and V. Sivaraman, "Verifying and Monitoring IoTs Network Behavior Using MUD Profiles," in IEEE Transactions on Dependable and Secure Computing, vol. 19, no. 1, pp. 1-18, 1 Jan.-Feb. 2022
```

**MUD Profile generation code**: https://github.com/ayyoob/mudgee

**Runtime Device identification**: https://github.com/ChakshuGupta/runtime-identification-mudgee


MUD profile generation tool called MUDgee is publically available at the link given above. We implemented the runtime device identification part of the code, which generates the MUD profiles from the runtime network traffic and tries to find a match with the known MUD profiles.


## Your Smart Home Can't Keep a Secret

**Paper**:
```
Shuaike Dong, Zhou Li, Di Tang, Jiongyi Chen, Menghan Sun, and Kehuan Zhang. 2020. Your Smart Home Can't Keep a Secret: Towards Automated Fingerprinting of IoT Traffic. In Proceedings of the 15th ACM Asia Conference on Computer and Communications Security (ASIA CCS '20).
```

**Code Repository**: https://github.com/ChakshuGupta/your-smart-home-can-t-keep-a-secret

We re-implemented the device identification method proposed in this paper.

## IoTDevID

**Paper**:
```
Kahraman Kostas, Mike Just, and Michael A. Lones. IoTDevID: A Behavior-Based Device Identification Method for the IoT, IEEE Internet of Things Journal, 2022.
```

**Original Code Repository**: https://github.com/kahramankostas/iotdevidv2

**Updated**: https://github.com/ChakshuGupta/IoTDevIDv2

The authors have made their code publicly available. However, since it was provided in Jupyter notebooks, we extracted the functions and consolidated them into a unified codebase for our evaluations.


## Devicemein

**Paper**:
```
Jorge Ortiz, Catherine Crawford, and Franck Le. 2019. DeviceMien: network device behavior modeling for identifying unknown IoT devices. In Proceedings of the International Conference on Internet of Things Design and Implementation (IoTDI '19).
```

**Code Repository**: https://github.com/ChakshuGupta/devicemien

We re-implemented the device identification method proposed in this paper.


## GenIoTID

**Paper**:
```
Maali, E., Alrawi, O. and McCann, J. (2025) ‘Evaluating machine learning-based IOT device identification models for security applications’, Proceedings 2025 Network and Distributed System Security Symposium [Preprint].
```

**Code Repository**: https://github.com/ChakshuGupta/geniotid

The paper evaluates existing methods and identifies the most generalizable feature set and ML algorithm for IoT device identification. We implemented this approach, GenIoTID, using the selected features and algorithm.