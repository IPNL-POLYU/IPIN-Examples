# User Guides

This directory contains user-facing guides and decision-making tools to help you use the IPIN-Examples repository effectively. These guides are intended for practitioners, researchers, and students who want to apply indoor positioning techniques.

---

## Available Guides

### Chapter 3: State Estimation

#### [`ch3_estimator_selection.md`](./ch3_estimator_selection.md)
**Comprehensive guide for choosing the right state estimator**

This 500+ line guide helps you select the appropriate estimator for your specific indoor positioning scenario.

**Contents:**
- Quick selection table (scenario → recommended estimator)
- Detailed analysis of each estimator (LS, KF, EKF, UKF, PF, FGO)
- When to use / pros / cons / complexity for each method
- Decision tree flowchart
- Accuracy vs speed trade-offs
- Common pitfalls and solutions
- Real performance data from examples
- Code examples for each estimator

**Target audience:**
- Practitioners selecting an estimator for their application
- Students learning about state estimation trade-offs
- Researchers comparing different methods

**Start here if:** You're asking "Which estimator should I use for my indoor positioning problem?"

---

## Coming Soon

Additional guides to be added:

- **Ch4 RF Positioning Guide** - Selecting TOA/TDOA/AOA methods
- **Ch5 Fingerprinting Guide** - Deterministic vs probabilistic methods
- **Ch6 Dead Reckoning Guide** - IMU calibration and integration best practices
- **Ch7 SLAM Guide** - When to use bundle adjustment vs pose graph optimization
- **Ch8 Sensor Fusion Guide** - Loosely coupled vs tightly coupled fusion

---

## How to Use These Guides

1. **Browse by chapter** - Find the guide for your topic of interest
2. **Use quick reference tables** - Get fast answers to common questions
3. **Read detailed sections** - Deep dive into specific methods
4. **Follow decision trees** - Systematic approach to selection
5. **Study code examples** - See how to implement in practice

---

## Related Resources

### For Theory and Equations
→ See chapter-specific documentation in `docs/`:
- `docs/ch2_equation_mapping.md` - Coordinate transformation equations
- `docs/CH2_QUICK_REFERENCE.md` - Quick reference for Ch2
- `docs/ch7_slam.md` - SLAM concepts and equations
- `docs/ch8_fusion_api_reference.md` - Sensor fusion API
- etc.

### For Implementation Details
→ See engineering documentation in [`docs/engineering/`](../engineering/)

### For Running Examples
→ See README files in each chapter directory:
- `ch2_coords/README.md`
- `ch3_estimators/README.md`
- `ch4_rf_point_positioning/README.md`
- `ch5_fingerprinting/README.md`
- `ch6_dead_reckoning/README.md`
- `ch7_slam/README.md`
- `ch8_sensor_fusion/README.md`

### For Notebooks and Interactive Learning
→ See `notebooks/` directory

---

## Contributing

To add a new user guide:

1. Focus on **practical decision-making** (not theory or implementation)
2. Include **quick reference tables** for fast lookup
3. Provide **real examples** and performance data
4. Use **clear sections** and navigation
5. Target **users/practitioners**, not developers
6. Add the guide to this README with a clear description

---

## Feedback

If you have suggestions for new guides or improvements to existing ones, please open an issue or submit a pull request.

---

**Note:** These are user-facing guides. For technical/engineering documentation, see the [`docs/engineering/`](../engineering/) directory.


