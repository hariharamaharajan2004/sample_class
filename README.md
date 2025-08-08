# AI-Powered Image Identification and Classification System

A comprehensive real-time object detection and classification system with incremental learning capabilities. The system uses YOLOv8 for object detection, pre-trained CNNs for feature extraction, and clustering algorithms for automatic class discovery.

## Features

### Core Capabilities
- **Real-time Object Detection**: YOLOv8-based detection for both live camera feeds and static images
- **Feature Extraction**: Pre-trained CNN models (MobileNet, EfficientNet, ResNet) for robust embeddings
- **Dynamic Classification**: Similarity-based matching with configurable thresholds
- **Automatic Class Discovery**: K-Means and DBSCAN clustering for unknown objects
- **Human-in-the-Loop Learning**: Manual review and labeling of discovered clusters
- **Incremental Learning**: Continuous improvement of the classification system

### Technical Features
- **Embedding-based Matching**: Cosine similarity for robust object classification
- **Local Database**: SQLite storage for embeddings, classes, and metadata
- **Web Interface**: Modern, responsive dashboard for system interaction
- **Camera Integration**: Live video stream processing with real-time annotations
- **Batch Processing**: Efficient handling of multiple images
- **Export/Import**: Data portability and backup capabilities

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Layer   │    │  Processing Core │    │  Storage Layer  │
│                 │    │                  │    │                 │
│ • Camera Feed   │───▶│ • Object Detector│───▶│ • SQLite DB     │
│ • Static Images │    │ • Feature Extract│    │ • Embeddings    │
│ • Batch Upload  │    │ • Classifier     │    │ • Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Discovery Layer │    │  Learning Core   │    │Interface Layer  │
│                 │    │                  │    │                 │
│ • Clustering    │◀───│ • Unknown Buffer │───▶│ • Web Dashboard │
│ • Class Creation│    │ • Similarity     │    │ • REST API      │
│ • Human Review  │    │ • Incremental    │    │ • CLI Interface │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Webcam (optional, for live camera features)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd sample_class
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the system:
```bash
python main.py web
```

## Usage

### Web Interface
Start the web application:
```bash
python main.py web --host 0.0.0.0 --port 12000
```

Access the dashboard at `http://localhost:12000`

#### Dashboard Features
- **System Statistics**: Real-time metrics and performance indicators
- **Image Upload**: Process single or multiple images
- **Live Camera**: Real-time object detection and classification
- **Class Management**: View, create, and manage object classes
- **Clustering Interface**: Discover and review new classes

### Command Line Interface

#### Process Images
```bash
# Process a single image
python main.py cli process path/to/image.jpg --save --output results.json

# Start camera processing
python main.py cli camera --source 0 --save
```

#### Manage Classes
```bash
# Add a new class manually
python main.py cli add-class "car" image1.jpg image2.jpg image3.jpg --description "Vehicle class"

# List all classes
python main.py cli list-classes

# Show system statistics
python main.py cli stats
```

#### Discover New Classes
```bash
# Discover classes using K-Means
python main.py cli discover --method kmeans

# Discover classes using DBSCAN
python main.py cli discover --method dbscan
```

## System Components

### 1. Object Detection (`src/core/object_detector.py`)
- **Model**: YOLOv8 (nano/small/medium variants)
- **Features**: Real-time detection, bounding box extraction
- **Output**: Object crops with confidence scores

### 2. Feature Extraction (`src/core/feature_extractor.py`)
- **Models**: MobileNet v3, EfficientNet B0, ResNet18
- **Features**: Normalized embeddings, similarity computation
- **Output**: 512-dimensional feature vectors

### 3. Classification Engine (`src/core/classification_engine.py`)
- **Similarity Threshold**: Configurable matching threshold
- **Unknown Buffer**: Automatic collection of unmatched objects
- **Incremental Learning**: Dynamic class updates

### 4. Clustering Manager (`src/clustering/cluster_manager.py`)
- **Algorithms**: K-Means, DBSCAN
- **Features**: Automatic cluster count determination, quality metrics
- **Output**: Cluster assignments and analysis

### 5. Database Manager (`src/database/db_manager.py`)
- **Storage**: SQLite with BLOB embeddings
- **Features**: ACID transactions, efficient queries
- **Schema**: Classes, embeddings, unknown objects, clusters

## Configuration

### Model Selection
Edit the initialization parameters in `main.py` or through the web interface:

```python
# Object Detection
detector = ObjectDetector(model_name='yolov8n.pt', confidence_threshold=0.5)

# Feature Extraction
feature_extractor = FeatureExtractor(model_name='mobilenet_v3_small', embedding_dim=512)

# Classification
engine = ClassificationEngine(
    similarity_threshold=0.7,
    unknown_buffer_size=50,
    clustering_interval=20
)
```

### Database Configuration
The system uses SQLite by default. For production deployments, consider:
- Regular database backups
- SSD storage for better performance
- Database optimization for large datasets

## Advanced Features

### Embedding-based Matching
The system uses cosine similarity for robust object matching:
- Normalized feature vectors ensure scale invariance
- Configurable similarity thresholds for precision/recall trade-offs
- Centroid-based class representation for efficiency

### Few-shot Learning
Support for creating classes with minimal examples:
- Minimum 1-3 sample images per class
- Automatic centroid computation
- Incremental updates as new samples are added

### Siamese Networks (Future Enhancement)
Planned implementation for improved similarity learning:
- Contrastive loss for better embeddings
- Triplet loss for metric learning
- Online hard negative mining

## Performance Optimization

### Hardware Recommendations
- **GPU**: NVIDIA GTX 1060+ for real-time processing
- **RAM**: 8GB+ for large datasets
- **Storage**: SSD for database operations

### Software Optimizations
- **Batch Processing**: Process multiple images simultaneously
- **Model Quantization**: Reduce model size for deployment
- **Caching**: Store frequently accessed embeddings in memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU processing
   - Use smaller model variants (yolov8n instead of yolov8l)

2. **Camera Access Issues**
   - Check camera permissions
   - Try different camera sources (0, 1, 2...)
   - Ensure camera is not used by other applications

3. **Slow Processing**
   - Enable GPU acceleration
   - Reduce image resolution
   - Use lighter model variants

4. **Database Errors**
   - Check disk space
   - Verify write permissions
   - Consider database maintenance

### Performance Monitoring
Monitor system performance through:
- Web dashboard statistics
- CLI stats command
- Database query performance
- Memory usage patterns

## Contributing

### Development Setup
1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
python -m pytest tests/
```

3. Code formatting:
```bash
black src/
flake8 src/
```

### Adding New Features
- Follow the existing architecture patterns
- Add comprehensive tests
- Update documentation
- Consider backward compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **scikit-learn**: Machine learning utilities
- **Flask**: Web framework

## Future Enhancements

### Planned Features
- [ ] Siamese network implementation
- [ ] Active learning strategies
- [ ] Multi-modal fusion (text + image)
- [ ] Distributed processing
- [ ] Cloud deployment options
- [ ] Mobile app interface
- [ ] Advanced visualization tools
- [ ] Model versioning and rollback
- [ ] Automated model retraining
- [ ] Integration with external APIs

### Research Directions
- Meta-learning for few-shot classification
- Continual learning without catastrophic forgetting
- Uncertainty quantification for predictions
- Adversarial robustness improvements
- Federated learning capabilities