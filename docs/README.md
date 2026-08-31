# Documentation

Welcome to the IPIN-Examples documentation. This directory contains various types of documentation to help you understand and use the code effectively.

---

## 📚 Documentation Types

### 🎯 [User Guides](./guides/)
**For practitioners, researchers, and students**

Decision-making tools and practical guides to help you apply indoor positioning techniques:

- **[Ch3 Estimator Selection Guide](./guides/ch3_estimator_selection.md)** - Comprehensive guide for choosing the right state estimator (LS, KF, EKF, UKF, PF, FGO)

→ *Start here if you're asking: "Which method should I use for my application?"*

---

### 🔧 [Engineering Documentation](./engineering/)
**For developers, maintainers, and contributors**

Technical implementation notes, bug fixes, and engineering decisions:

- **[Complete Implementation Summary](./engineering/complete_implementation_summary.md)** - Master overview of ch3 improvements
- **[Ch3 Production Fixes](./engineering/ch3_production_fixes.md)** - Critical fixes (angle wrapping, singularity handling, observability)
- **[Ch3 Robustness Improvements](./engineering/ch3_robustness_improvements.md)** - Input validation, shared models, tests
- **[Ch3 Bugfix Summary](./engineering/ch3_bugfix_summary.md)** - Robust LS fix details

→ *Read these if you're contributing code or need to understand implementation details*

---

### 📖 Chapter-Specific References
**Technical references and equation mappings**

Detailed documentation for each chapter of the book:

- **[ch2_equation_mapping.md](./ch2_equation_mapping.md)** - Coordinate transformation equations
- **[CH2_QUICK_REFERENCE.md](./CH2_QUICK_REFERENCE.md)** - Quick reference for Ch2
- **[ch7_slam.md](./ch7_slam.md)** - SLAM concepts and equations
- **[ch8_fusion_api_reference.md](./ch8_fusion_api_reference.md)** - Sensor fusion API reference
- **[ch8_lc_tc_comparison_guide.md](./ch8_lc_tc_comparison_guide.md)** - Loosely vs tightly coupled fusion comparison

---

### 🗂️ Additional Documentation

- **[data_simulation_guide.md](./data_simulation_guide.md)** - How to generate and use simulated datasets
- **[equation_index.yml](./equation_index.yml)** - Searchable index of equations

---

## 🧭 Quick Navigation

**I want to...**

| Goal | Where to Go |
|------|-------------|
| Choose which estimator to use | [`guides/ch3_estimator_selection.md`](./guides/ch3_estimator_selection.md) |
| Understand a code implementation | [`engineering/`](./engineering/) directory |
| Map equations to code | Chapter-specific docs (e.g., `ch2_equation_mapping.md`) |
| Generate datasets | [`data_simulation_guide.md`](./data_simulation_guide.md) |
| Run examples | See README files in each `ch*_*/` folder |
| Understand the book structure | Main [`README.md`](../README.md) at repository root |

---

## 📝 Documentation Structure

```
docs/
├── README.md                          # This file (navigation hub)
│
├── guides/                            # User-facing guides
│   ├── README.md                      # Guide index
│   └── ch3_estimator_selection.md     # Estimator selection guide
│
├── engineering/                       # Technical/engineering docs
│   ├── README.md                      # Engineering doc index
│   ├── complete_implementation_summary.md
│   ├── ch3_implementation_summary.md
│   ├── ch3_production_fixes.md
│   ├── ch3_robustness_improvements.md
│   └── ch3_bugfix_summary.md
│
└── [chapter-specific docs]            # Ch2, Ch7, Ch8 references
    ├── ch2_equation_mapping.md
    ├── CH2_QUICK_REFERENCE.md
    ├── ch7_slam.md
    ├── ch8_fusion_api_reference.md
    ├── ch8_lc_tc_comparison_guide.md
    ├── data_simulation_guide.md
    └── equation_index.yml
```

---

## 🤝 Contributing Documentation

When adding new documentation:

### For User Guides (`guides/`)
- Focus on **practical decision-making**
- Include **quick reference tables**
- Provide **real examples** and performance data
- Target **users/practitioners**

### For Engineering Docs (`engineering/`)
- Focus on **implementation details**
- Explain **why** decisions were made
- Document **bugs** and their fixes
- Target **developers/maintainers**

### For Chapter References
- Map **equations to code**
- Provide **API references**
- Include **usage examples**
- Maintain **consistency** with book notation

---

## 📚 Related Resources

- **Main README:** [`../README.md`](../README.md) - Repository overview and setup
- **Jupyter Notebooks:** [`../notebooks/`](../notebooks/) - Interactive tutorials
- **Chapter READMEs:** `../ch*_*/README.md` - Chapter-specific examples and usage
- **Tests:** [`../tests/`](../tests/) - Unit tests showing API usage

---

**Questions or suggestions?** Open an issue or submit a pull request!



