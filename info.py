# 1. ประเภทของ ML  ?
#    - ดูจาก Input เป็น Supervised Learning เพราะข้อมูลมี labeled
#    - ดูจาก Output เป็น Classification problem เพราะ output ที่ได้เป็น class
# 2. ทำความเข้าใขกับข้อมูล
#    - Analyze the Data
#       - สถิติเชิงพรรณนา
#       - สร้างกราฟ Data Visualization
#    - Process the Data
#       - pre-processing (Feature Extraction) คัดเลือก Data ที่จะนำมาเป็น feature ตัด data ไม่ใช้หรือไม่มีประโยชน์ออกให้หมด
#       - profiling
#       - cleansing การตรวจสอบความถูกต้องของข้อมูลทั้งหมด
#    - Transform the data
#       - การแปลงข้อมูลดิบไปเป็นข้อมูลที่เราจะนำมาใช้สอน ml
# 3. หาอัลกอลิทึมที่ใช้ได้ดีกับข้อมูล
#    -  ตัดสินใจจาก
#       - The accuracy of the model  ความเเม่นยำของ model
#       - The interpretability of the model  ความสัมพันธ์กันของ data กับ model
#       - The complexity of the model  ความซับซ้อนของ model
#       - The scalability of the model  ความสามารถในการปรับขนาดของ model
#       - How long does it take to build, train, and test the model?  เวลาที่ใช้ไปในการสร้างเเละสอน model
#       - How long does it take to make predictions using the model?  เวลาที่ใช้ไปในทำนายของ model
#       - Does the model meet the business goal?  model ตอบโจทย์กับธุรกิจหรือป่าว
# 4. นำ machine learning มาใช้งาน
#    -  machine learning pipeline
# 5. เพิ่มประสทิธิภาพให้กับ Hyperparameter
#    - มีสามวิธีคือ grid search, random search, Bayesian optimization
#
# Machine Learning Task
#   - Supervised learning
#       - Regression
#       - Classification
#   - Unsupervused learning
#       - Cluster
#   - Reinforcement learning
#
# Machine Learning Algorithms
#   - Linear Regression
#       - ข้อมูลเชิงปริมาณ quantitative
#       - หาความสัมพันธ์ของตัวแปร X ที่เป็นตัวแปรอิสระกับ Y ที่เป็นตัวแปรตาม
#       - loss function ด้วย mean squared error (MSE) , mean absolute error (MAE)
#   - Logistic Regression
#       - เป็นประเภท classification เป็น subset ของ supervised learning
#       - output เป็น categorical หรือ binary
#   - K-means
#       - เป็นประเภท clustering เป็น subset ของ unsupervised learning
#       - จัดกลุ่มข้อมูลที่เข้ามาว่าจะอยู่ในกลุ่มไหนตามความเหมือนหรือคล้ายกัน
#       - K คือจำนวน cluster
#   - K-nearest-neigbors
#       - เป็นประเภท classification เป็น subset ของ supervised learning
#       - เป็นการจัดกลุ่มข้อมูลเมื่อมี data ใหม่เข้ามา data มีค่าใกล้เคียงกับค่าใดก็จะอยู่กลุ่มนั้น
#       - K คือค่าที่ใช้ในการหนดการ Vote
#   - Support Vector Machines
#       - เป็นประเภท classification
#       - เเบ่งเป็น 2 class ด้วยเส้นตรง
#   - Random Forest
#       - เป็นประเภท regression,classification ป็น subset ของ Supervised learning
#       - Rule base คือต้องสร้างกฏ if-else จาก feature ซิ่งมา decision tree หลายๆ อัน
#       - เป็น model แบบ ensemble คือใช้ model หลายๆ model มาประกอบกันเป็น model ที่ซับซ้อน
#   - Neural networks
