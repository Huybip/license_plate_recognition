# license_plate_recognition
Tóm tắt quá trình thực hiện
Quá trình phát triển hệ thống nhận diện biển số xe tự động được triển khai qua các bước chính sau:
Bước 1: Tiền xử lý dữ liệu
Thu thập tập dữ liệu biển số xe thực tế (ảnh chụp từ camera giám sát).


Gán nhãn cho dữ liệu bằng công cụ hỗ trợ (Roboflow, LabelImg).


Chuyển đổi định dạng ảnh và nhãn để huấn luyện mô hình (YOLO format).


Bước 2: Phát hiện biển số
Sử dụng mô hình YOLOv8 để phát hiện vùng chứa biển số trong ảnh đầu vào.


Kết quả đầu ra là bounding box xác định chính xác vùng biển số cần trích xuất.


Bước 3: Cắt và chuẩn hóa ảnh biển số
Cắt ảnh theo bounding box từ YOLO.


Chuẩn hóa ảnh đầu vào để phù hợp với đầu vào của các mô hình OCR: chuyển về thang xám, tăng cường độ tương phản.


Bước 4: Nhận diện ký tự bằng OCR
Áp dụng 2 phương pháp:


EasyOCR (dựa trên CRNN + CTC): cho kết quả nhanh, hỗ trợ nhiều ngôn ngữ.


Tesseract (dựa trên LSTM): mã nguồn mở của Google, hoạt động tốt với biển số rõ nét.


Bước 5: Hiển thị và đánh giá kết quả
Hiển thị kết quả nhận dạng chồng lên ảnh gốc.


So sánh với ground truth để tính các chỉ số như độ chính xác, độ phủ, tốc độ xử lý.
MỞ ĐẦU 
Lý do chọn đề tài 
Trong thời đại công nghệ số phát triển mạnh mẽ, các thành phố đang dần hướng tới mô hình giao thông thông minh và tự động hóa. Một trong những ứng dụng nổi bật và thực tiễn nhất của thị giác máy tính trong lĩnh vực này chính là nhận diện biển số xe – bài toán có vai trò then chốt trong các hệ thống giám sát phương tiện, kiểm soát ra vào, thu phí tự động, truy vết vi phạm giao thông, và hỗ trợ điều tra hình sự.
Tại Việt Nam, biển số xe có nhiều kiểu định dạng khác nhau về phông chữ, màu nền, độ nghiêng và chất lượng ảnh đầu vào, đòi hỏi hệ thống nhận diện phải xử lý tốt nhiều khâu: từ tiền xử lý ảnh, phát hiện vùng chứa đối tượng cho đến nhận dạng ký tự quang học (OCR). Đây là một bài toán tổng hợp, có độ phức tạp nhất định, nhưng đồng thời cũng rất lý tưởng để sinh viên vận dụng kiến thức về xử lý ảnh, phát hiện đối tượng và nhận dạng ký tự trong một quy trình hoàn chỉnh.
Do đó, để tiếp cận và thực hành bài toán ứng dụng có tính tích hợp cao giữa các kỹ thuật xử lý ảnh truyền thống và công cụ nhận dạng ký tự hiện đại, nhóm chúng em lựa chọn đề tài “Nhận diện biển số xe” làm nội dung bài tập lớn của học phần Thị giác máy tính.
Mục tiêu của đề tài 
Đề tài hướng tới việc xây dựng một mô hình nhận diện biển số xe đơn giản nhưng hiệu quả, có thể xử lý ảnh tĩnh đầu vào và trả về kết quả ký tự nhận dạng được từ biển số. Cụ thể:
Ứng dụng các kỹ thuật xử lý ảnh để xác định và trích xuất vùng chứa biển số từ ảnh gốc;


Áp dụng mô hình OCR có sẵn (EasyOCR, Tesseract) để nhận diện nội dung biển số;


Kiểm thử hệ thống với một tập ảnh thực tế nhằm đánh giá hiệu quả nhận dạng;


Rút ra bài học, đánh giá ưu điểm, hạn chế và định hướng mở rộng hệ thống trong tương lai.
Đối tượng và phạm vi nghiên cứu 
Đối tượng nghiên cứu: Ảnh xe có chứa biển số rõ nét, không bị che khuất, chụp trong điều kiện ánh sáng ban ngày.
Phạm vi thực hiện:
Chỉ xử lý ảnh tĩnh, không xét đến ảnh động hoặc video;


Không tự huấn luyện mô hình nhận dạng, mà sử dụng mô hình OCR đã huấn luyện sẵn;


Tập trung vào quy trình phát hiện và nhận dạng, không triển khai giao diện hay hệ thống phần mềm hoàn chỉnh.
Ý nghĩa của đề tài 
Đề tài là cơ hội để vận dụng tổng hợp kiến thức về xử lý ảnh số, phát hiện đối tượng, và nhận dạng ký tự – những thành phần cốt lõi trong lĩnh vực thị giác máy tính. Đồng thời, đề tài cũng rèn luyện kỹ năng làm việc với các thư viện mã nguồn mở phổ biến hiện nay, chẳng hạn như OpenCV và EasyOCR. Về mặt thực tiễn, mô hình xây dựng trong đề tài là một bước đầu đơn giản nhưng có tiềm năng ứng dụng trong các hệ thống giám sát phương tiện, quản lý bãi đỗ xe, hoặc làm nền tảng cho các nghiên cứu sâu hơn về xử lý ảnh giao thông.







CHƯƠNG 1: GIỚI THIỆU
1.1. Đặt vấn đề 
Thị giác máy tính( Computer Vision) là một lĩnh vực trọng yếu trong trí tuệ nhân tạo, cho phép máy tính “nhìn thấy” và hiểu đươc hình ảnh hoặc video như con người. Trong những năm gần đây, cùng với sự phát triển mạnh mẽ của phần cứng và các thư viện mã nguồn mở, thị giác máy tính đã được ứng dụng ngày càng rộng rãi trong các lĩnh vực như y tế, nông nghiệp, công nghiệp và đặc biệt là giao thông thông minh.

Trong số đó, bài toán nhận diện biển số xe (Automatic License Plate Recognition - ALPR) được xem là một trong những ứng dụng điển hình và có tính thực tiễn cao. Hệ thống nhận diện biển số xe thường được tích hợp trong các camera giám sát tại bãi đỗ xe, trạm thu phí, kiểm soát an ninh hoặc các nút giao thông để thực hiện các chức năng như: kiểm soát ra vào, thu phí tự động, phát hiện xe vi phạm, truy vết phương tiện. v.v..

Tuy nhiên, việc nhận diện chính xác biển số xe trong điều kiện thực tế không hề đơn giản, do ảnh hưởng của nhiều yếu tố như: ánh sáng, góc chụp, độ phân giải ảnh, chất lượng biển số, phông chữ đa dạng hoặc tình trạng che khuất một phần biển. Vì vậy, cần có sự kết hợp hiệu quả giữa các bước tiền xử lý ảnh, phát hiện đối tượng (vùng chứa biển số), và nhận dạng ký tự thông qua OCR.
1.2. Các công trình, sản phẩm đã tồn tại 
Trong những năm gần đây, bài toán nhận diện biển số xe (ALPR – Automatic License Plate Recognition) đã thu hút nhiều sự quan tâm từ cộng đồng nghiên cứu cũng như các doanh nghiệp công nghệ. Trên thế giới, nhiều sản phẩm thương mại và hệ thống mã nguồn mở đã được phát triển với độ chính xác cao và khả năng hoạt động trong thời gian thực.
Một số hệ thống điển hình có thể kể đến như:
OpenALPR (nay là Plate Recognizer): một dịch vụ ALPR dựa trên đám mây, hỗ trợ nhận diện biển số của hơn 100 quốc gia, bao gồm cả Việt Nam. Hệ thống sử dụng mạng nơ-ron tích chập (CNN) để phát hiện vùng biển số, sau đó áp dụng mô hình nhận dạng ký tự để trích xuất văn bản từ biển số. Đây là giải pháp thương mại được tích hợp vào các hệ thống quản lý bãi xe, trạm thu phí tại Mỹ, Nhật, và nhiều quốc gia châu Âu.


EasyOCR: là một thư viện mã nguồn mở được phát triển bởi Jaided AI, cho phép nhận dạng văn bản trong ảnh với hơn 80 ngôn ngữ. Thư viện này không yêu cầu huấn luyện lại, hỗ trợ cả tiếng Việt và hoạt động tốt trong các ứng dụng nhận dạng biển số xe nếu vùng chứa biển được cắt chính xác. Đây chính là thư viện được nhóm sử dụng trong bài tập lớn này.


YOLO (You Only Look Once): là một trong những mô hình phát hiện đối tượng thời gian thực nổi tiếng nhất hiện nay. Nhiều dự án ALPR hiện đại kết hợp YOLO (để phát hiện vùng biển số) với CRNN hoặc Tesseract (để nhận dạng ký tự). Ví dụ: trong nghiên cứu của nhóm tác giả Hegazy et al. (2021), hệ thống kết hợp YOLOv3 và LSTM để nhận diện biển số Ai Cập với độ chính xác >90% trên ảnh thực tế.


Deep License Plate Recognition (D-LPR): Là mô hình học sâu kết hợp ResNet để phát hiện và nhận dạng biển số ở điều kiện phức tạp như bị nghiêng, mờ, hoặc biển số bị che một phần. Một số ứng dụng tại Hàn Quốc và Trung Quốc đã triển khai hệ thống tương tự trên camera giám sát đường phố.


Tại Việt Nam, một số đơn vị đã triển khai các hệ thống nhận diện biển số xe thực tế:
Ứng dụng tại các tòa nhà, bãi xe thông minh (VD: eParking, Smart Parking) sử dụng camera AI kết hợp phần mềm nhận diện để tự động ghi nhận biển số khi xe ra vào.


Nghiên cứu tại Đại học Bách Khoa TP.HCM đã đề xuất mô hình nhận diện biển số sử dụng YOLOv4 kết hợp CRNN, huấn luyện trên tập dữ liệu biển số Việt Nam thu thập từ thực tế giao thông.
CHƯƠNG 2: PHƯƠNG PHÁP NHẬN DIỆN BIỂN SỐ XE 
2.1. Tổng quan quy trình

Hệ thống nhận diện biển số xe được xây dựng theo mô hình pipeline xử lý tuần tự, trong đó từng bước được thiết kế nhằm tối ưu độ chính xác trong việc trích xuất và nhận dạng nội dung biển số từ ảnh hoặc video đầu vào. Cấu trúc tổng thể bao gồm bốn giai đoạn chính, được mô tả như sau:
2.1.1.  Tiền xử lý ảnh và dữ liệu đầu vào
Dữ liệu đầu vào có thể là ảnh tĩnh hoặc khung hình trích từ video. Mỗi ảnh sẽ được chuẩn hóa thông qua các bước xử lý cơ bản:
Chuyển đổi từ ảnh màu sang ảnh xám (grayscale) để giảm chiều dữ liệu.


Cân bằng sáng và tăng cường độ tương phản để làm nổi bật thông tin cần thiết.


Khử nhiễu bằng bộ lọc Gaussian nhằm làm mượt các cạnh và giảm chi tiết không mong muốn.
2.1.2. Phát hiện vùng chứa biển số bằng mô hình YOLOv8
Sau bước tiền xử lý, ảnh được đưa vào mô hình YOLOv8 (You Only Look Once phiên bản 8) – một trong những mô hình phát hiện đối tượng tiên tiến, có khả năng hoạt động thời gian thực. Mô hình phân tích ảnh và đưa ra các bounding box (hộp giới hạn) tương ứng với vùng được dự đoán là biển số xe.
2.1.3. Trích xuất và chuẩn hóa vùng biển số
Từ các bounding box được xác định, hệ thống tiến hành cắt (crop) vùng ảnh tương ứng từ ảnh gốc. Vùng ảnh biển số sau đó được xử lý thêm để chuẩn hóa:
Căn chỉnh kích thước về chuẩn đầu vào cho OCR.


Làm mượt ảnh, tăng độ nét các ký tự.


Nhị phân hóa (binarization) nhằm tách biệt rõ ràng vùng ký tự với nền.
2.1.4. Nhận dạng ký tự bằng công nghệ OCR
Vùng biển số đã chuẩn hóa được đưa vào song song hai công cụ OCR:
EasyOCR – thư viện học sâu mã nguồn mở hỗ trợ tiếng Việt và các ký tự Latin.


Tesseract – công cụ OCR truyền thống, được sử dụng như phương án bổ sung để tăng tính chính xác.


Kết quả nhận dạng được trả về dưới dạng chuỗi văn bản và có thể hiển thị trực tiếp lên ảnh đầu ra, hoặc lưu vào hệ thống để xử lý tiếp theo.
Sơ đồ pipeline tổng quát
                  
               hình 2.1.4.1: sơ đồ pipeline tổng quát


Sơ đồ kiến trúc tổng quát giúp đảm bảo việc xử lý theo tuần tự logic và dễ dàng tích hợp thêm bước đánh giá hiệu suất, hiển thị thời gian thực hoặc ghi log phục vụ phân tích sau này.
Đặc điểm triển khai
Ngôn ngữ lập trình: Python


Môi trường: Google Colaboratory


Thư viện sử dụng: OpenCV, Ultralytics, EasyOCR, pytesseract, matplotlib


Hệ thống hỗ trợ mở rộng với các module thời gian thực, đánh giá hiệu suất hoặc lưu log phục vụ kiểm thử, giám sát.
2.2. YOLOv8 – Nhận diện biển số xe
2.2.1. Tổng quan về YOLO
YOLO (You Only Look Once) là một trong những kiến trúc tiên tiến và hiệu quả nhất trong lĩnh vực phát hiện đối tượng theo thời gian thực. Từ phiên bản đầu tiên (YOLOv1) đến hiện tại (YOLOv8), mô hình đã trải qua nhiều cải tiến đáng kể về kiến trúc mạng, tốc độ và độ chính xác. YOLOv8 – phiên bản mới nhất do Ultralytics phát hành – có khả năng phát hiện nhanh, chính xác và hoạt động ổn định cả trên GPU lẫn CPU, phù hợp với nhiều ứng dụng từ nghiên cứu đến triển khai thực tế.
2.2.2. Nguyên lý hoạt động
YOLOv8 chia ảnh đầu vào thành nhiều ô lưới và gán trách nhiệm phát hiện cho các ô có chứa tâm đối tượng. Mô hình học các đặc trưng (features) thông qua mạng tích chập sâu (deep CNN), sau đó dự đoán:
Tọa độ bounding box: x,y,w,hx, y, w, hx,y,w,h (trung tâm, chiều rộng và chiều cao)


Xác suất tồn tại đối tượng trong ô


Xác suất phân lớp (class probability) của đối tượng


2.2.3. Ứng dụng trong đề tài
Trong khuôn khổ đề tài, YOLOv8 được sử dụng để phát hiện vùng chứa biển số xe trong ảnh đầu vào. Mô hình có thể được huấn luyện hoặc tinh chỉnh (fine-tune) trên bộ dữ liệu biển số Việt Nam theo định dạng YOLO. Kết quả đầu ra là các bounding box được sử dụng để trích xuất vùng biển số, phục vụ cho bước nhận dạng ký tự bằng OCR ở các giai đoạn tiếp theo.
2.3. Tiền xử lý ảnh
Sau khi cắt vùng chứa biển số từ bounding box, ảnh cần được xử lý tiền xử lý để nâng cao hiệu quả nhận dạng ký tự. Các bước xử lý bao gồm:
Chuyển sang ảnh xám (grayscale): giảm chiều dữ liệu và loại bỏ thông tin màu không cần thiết.


Tăng cường độ tương phản: bằng các phương pháp như contrast stretching hoặc CLAHE để làm nổi bật ký tự.


Làm mịn ảnh (Gaussian Blur): giúp giảm nhiễu, làm trơn vùng ký tự.


Nhị phân hóa (adaptive thresholding): chuyển ảnh về đen trắng rõ ràng, tách ký tự ra khỏi nền.


Chuẩn hóa kích thước ảnh: đảm bảo đồng đều đầu vào cho các mô hình OCR.


Tùy theo phương pháp nhận dạng (EasyOCR hoặc Tesseract), quy trình tiền xử lý có thể được điều chỉnh linh hoạt.
2.4 EasyOCR – Nhận dạng ký tự bằng deep learning
EasyOCR là một thư viện mã nguồn mở nổi bật trong lĩnh vực nhận dạng ký tự quang học (OCR), được phát triển dựa trên mô hình học sâu CRNN (Convolutional Recurrent Neural Network). Đây là lựa chọn phổ biến trong các ứng dụng thực tế vì khả năng nhận diện chính xác cả ký tự Latin và phi Latin, bao gồm cả tiếng Việt.
2.4.1. Mô hình CRNN trong EasyOCR
CRNN là sự kết hợp giữa mạng tích chập (CNN) để trích xuất đặc trưng không gian từ ảnh và mạng hồi tiếp (RNN) để xử lý chuỗi ký tự liên tục. Cấu trúc tổng thể của EasyOCR bao gồm:
CNN backbone: trích xuất feature map từ ảnh đầu vào.
Bi-LSTM: học mối quan hệ chuỗi giữa các đặc trưng.
CTC Loss (Connectionist Temporal Classification): ánh xạ feature thành chuỗi ký tự không cần căn chỉnh ký tự thủ công.
2.4.2. Tổng quan về CRNN
2.4.2.1. Giới thiệu ngắn về CRNN:
CRNN (Convolutional Recurrent Neural Network) là mô hình kết hợp CNN và RNN. CNN giúp trích xuất đặc trưng từ ảnh đầu vào, sau đó RNN xử lý chuỗi đặc trưng theo chiều ngang (dòng ký tự), cho phép mô hình học được mối liên kết giữa các ký tự liên tiếp.
2.4.2.2. Vai trò của CTC (Connectionist Temporal Classification):
CTC là hàm mất mát đặc biệt dùng để huấn luyện các mô hình nhận dạng chuỗi mà không cần phải gán nhãn từng ký tự cụ thể tại vị trí chính xác. Trong EasyOCR, CTC giúp mô hình ánh xạ từ chuỗi đặc trưng trích xuất được sang chuỗi ký tự đầu ra mà không cần căn chỉnh thủ công.
2.4.2.3. Sơ đồ mô tả pipeline:

                            hình 2.4.2.1: sơ đồ minh họa quá trình crnn
2.4.3. Ưu điểm nổi bật
Hỗ trợ hơn 80 ngôn ngữ, bao gồm tiếng Việt.
Dễ tích hợp, không cần huấn luyện lại với ảnh thông thường.
Xử lý tốt ảnh nhiễu, mờ, chữ nghiêng hoặc không căn lề.
Có thể chạy trên cả CPU và GPU.
2.4.4. Ứng dụng trong đề tài
Trong đề tài này, EasyOCR được sử dụng để đọc chuỗi ký tự trong vùng biển số đã được phát hiện và cắt bởi YOLOv8. Trình tự xử lý:
Đầu vào là ảnh biển số được crop từ ảnh gốc.
Tiền xử lý: chuyển grayscale, resize, threshold.
Gọi hàm easyocr.Reader() với ngôn ngữ 'vi' để nhận dạng ký tự.
Kết quả gồm chuỗi ký tự và độ tin cậy (confidence).
Ví dụ minh họa:
Ảnh biển số đầu vào: https://drive.google.com/drive/folders/10y-oBJT3AGoUEoRWl2bCraDUXqPr1zPh?usp=drive_link

EasyOCR trả về: ví dụ:  ['Biển số nhận diện: 29-M1 253.99', confidence: 0.96]
2.4.5. Minh họa trực quan
Hình minh họa dưới đây thể hiện quá trình nhận diện ký tự bằng EasyOCR:
[Hình ảnh: ảnh biển số sau khi crop → kết quả hiển thị chuỗi ký tự] (các hình ảnh ở trong phần phụ lục A)

                hình 2.4.5.1. Kết quả sau khi nhận diện được ra biển số
2.4.6. Đánh giá
EasyOCR hoạt động tốt với các ảnh biển số thực tế có chất lượng trung bình đến cao. Nó khắc phục được nhiều hạn chế mà các hệ thống OCR truyền thống gặp phải trong môi trường không lý tưởng như ánh sáng yếu, biển số nghiêng, chữ bị mờ hoặc font không chuẩn. Điều này khiến EasyOCR trở thành lựa chọn tối ưu cho các bài toán thực tế như trong đề tài này. EasyOCR là thư viện OCR mã nguồn mở mạnh mẽ sử dụng mô hình CRNN (Convolutional Recurrent Neural Network) kết hợp với CTC loss để nhận dạng chuỗi ký tự trong ảnh. Hỗ trợ hơn 80 ngôn ngữ, trong đó có tiếng Việt.
Ưu điểm:
Không cần huấn luyện lại với các ảnh thông thường.
Nhận diện tốt ảnh phức tạp (nền không đồng nhất, bị nghiêng).
Có thể hoạt động tốt trên GPU và CPU.
Trong đề tài, EasyOCR được sử dụng để nhận dạng vùng ảnh biển số đã được YOLO crop và tiền xử lý.
2.5. Nhận diện biển số xe bằng Tesseract OCR
Trong hệ thống nhận diện biển số xe, bên cạnh EasyOCR, nhóm cũng tiến hành tích hợp và thử nghiệm công cụ Tesseract OCR nhằm so sánh hiệu quả và độ chính xác trong quá trình trích xuất văn bản từ vùng biển số đã được phát hiện.
2.5.1 Giới thiệu Tesseract
Tesseract là một công cụ nhận diện ký tự quang học (OCR) mã nguồn mở, ban đầu được phát triển bởi HP và hiện tại được duy trì bởi Google. Tesseract hỗ trợ nhiều ngôn ngữ và có thể nhận dạng ký tự từ ảnh văn bản, đặc biệt là các ảnh có độ tương phản tốt và ký tự rõ ràng. Công cụ này thường được sử dụng trong các ứng dụng chuyển đổi ảnh sang văn bản, xử lý tài liệu số, hoặc hệ thống tự động hóa thông tin từ ảnh chụp.
Tesseract hoạt động hiệu quả nhất với các ảnh đầu vào đã qua tiền xử lý như chuyển sang ảnh xám, nhị phân hóa, hoặc lọc nhiễu để làm nổi bật các ký tự cần trích xuất.
2.5.2 Tổng quan về mô hình LSTM trong Tesseract
Từ phiên bản 4.0, Tesseract đã tích hợp mô hình học sâu sử dụng mạng LSTM (Long Short-Term Memory) – một loại mạng nơ-ron hồi tiếp (Recurrent Neural Network) có khả năng ghi nhớ thông tin dài hạn và xử lý chuỗi dữ liệu.
Trong Tesseract, LSTM cho phép mô hình xử lý cả chuỗi ký tự liên tiếp trên ảnh thay vì nhận dạng từng ký tự riêng lẻ. Điều này làm tăng khả năng phân tích ngữ cảnh, giảm sai sót do nhiễu hoặc biến dạng và cải thiện độ chính xác chung. Cụ thể:
LSTM giúp mô hình “nhớ” cấu trúc ký tự liền nhau như biển số “30F-123.45”.


Hạn chế việc nhầm lẫn giữa các ký tự tương tự như ‘O’ và ‘0’, ‘I’ và ‘1’.


Hỗ trợ nhận dạng tốt hơn trên ảnh nghiêng nhẹ hoặc có ký tự dính liền.
2.5.3 Ưu - nhược điểm của Tesseract
Tesseract có nhiều ưu điểm khiến nó trở thành lựa chọn phù hợp cho các hệ thống OCR:
Mã nguồn mở và miễn phí: Dễ dàng tích hợp vào các hệ thống phần mềm, đặc biệt là trong nghiên cứu và học thuật.


Hỗ trợ nhiều ngôn ngữ và định dạng đầu vào.


Độ chính xác cao trên ảnh rõ ràng: Tesseract hoạt động hiệu quả với ảnh có độ tương phản tốt và phông chữ đều.


Không cần huấn luyện lại: Có thể sử dụng ngay với các mô hình tích hợp sẵn, phù hợp với biển số thông thường.


Tuy nhiên, Tesseract cũng có một số hạn chế nhất định, đặc biệt là:
Hiệu quả giảm rõ rệt khi ảnh mờ, biển số nghiêng mạnh hoặc có nhiễu nền.


Không có cơ chế tự động căn chỉnh vùng nhận dạng nếu ảnh bị xoay.


2.5.4 Ứng dụng trong đề tài
Trong đề tài này, nhóm tiến hành trích xuất vùng chứa biển số xe bằng mô hình YOLOv8, sau đó truyền ảnh cắt vùng biển số vào Tesseract để nhận diện chuỗi ký tự.
Tesseract được sử dụng như một lựa chọn so sánh với EasyOCR, nhằm đánh giá khả năng nhận diện trên cùng một tập dữ liệu. Kết quả nhận dạng sẽ được lưu trữ và hiển thị cùng với ảnh gốc trong video hoặc ảnh đầu vào.

Tesseract được sử dụng như một lựa chọn so sánh với EasyOCR, nhằm đánh giá khả năng nhận diện trên cùng một tập dữ liệu. Kết quả nhận dạng sẽ được lưu trữ và hiển thị cùng với ảnh gốc trong video hoặc ảnh đầu vào.
2.5.5 Đánh giá kết quả
Sau khi thử nghiệm trên các đoạn video và ảnh biển số trong tập kiểm thử, nhóm nhận thấy:
Với các ảnh biển số rõ ràng, phông chữ chuẩn, Tesseract cho kết quả rất tốt, gần như chính xác tuyệt đối.


Tuy nhiên, trong các trường hợp biển số nghiêng, ánh sáng yếu hoặc bị mờ, kết quả nhận diện của Tesseract bị giảm mạnh, thậm chí nhận sai hoàn toàn.


So với EasyOCR, Tesseract có tốc độ xử lý nhanh hơn, nhưng khả năng chịu lỗi kém hơn, đặc biệt không tốt với ảnh thực tế ngoài trời (biển số bị che, bị xước, bóng đổ).


Từ những quan sát này, nhóm kết luận rằng Tesseract phù hợp cho các hệ thống nhận diện ảnh rõ nét hoặc xử lý trong điều kiện tiêu chuẩn, còn đối với ứng dụng ngoài thực tế, EasyOCR tỏ ra linh hoạt và chính xác hơn.
2.6. So sánh tổng quan EasyOCR và Tesseract

Tiêu chí 
EasyOCR
Tesseract
Kiểu mô hình 
CRNN (Deep learning)
Heuristic + LSTM
Ngôn ngữ hỗ trợ
Hơn 80 (có tiếng Việt)
Hơn 100
Khả năng tùy biến 
Cao, dễ tinh chỉnh
Hạn chế
Yêu cầu tiền xử lý
Trung bình
Cao
Độ chính xác
Tốt với ảnh thực tế 
Tốt nếu ảnh rõ nét
Tốc độ xử lý
Trung bình
Nhanh

                                
                    bảng 2: so sánh tổng quan giữa EasyOCR và Tesseract
Kết luận:
 EasyOCR phù hợp hơn với ảnh biển số thực tế có nhiễu, biến dạng, trong khi Tesseract cho kết quả tốt hơn trong điều kiện ảnh đầu vào đã được xử lý sạch và rõ nét.


CHƯƠNG 3: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ
3.1 Môi trường triển khai
Trong quá trình phát triển hệ thống nhận diện biển số xe, việc lựa chọn môi trường phù hợp để lập trình, thử nghiệm và triển khai là yếu tố vô cùng quan trọng. Nhằm đảm bảo tính linh hoạt, dễ truy cập và tiết kiệm chi phí phần cứng, nhóm đã lựa chọn nền tảng Google Colab – một công cụ mạnh mẽ cho phép sử dụng GPU miễn phí và tích hợp sẵn nhiều thư viện phục vụ cho lập trình Python và học sâu.
Hệ điều hành: Linux (trên nền tảng máy chủ của Google Colab)
Ngôn ngữ lập trình: Python 3.10+
Các thư viện sử dụng:
Ultralytics: dùng để tải và triển khai mô hình YOLOv8
OpenCV: xử lý ảnh và video
EasyOCR: nhận dạng ký tự bằng deep learning
Pytesseract: sử dụng thư viện Tesseract của Google cho OCR
Matplotlib: hiển thị ảnh và kết quả trực quan
Phần cứng: GPU NVIDIA T4 (qua Colab), RAM 12GB, CPU 2 nhân
Lệnh cài đặt các thư viện:
!pip install ultralytics easyocr opencv-python pytesseract matplotlib
Việc triển khai trên Google Colab giúp giảm thiểu yêu cầu cấu hình máy tính cá nhân, đồng thời dễ dàng chia sẻ notebook và cùng cộng tác trong nhóm.
3.2 Thử nghiệm với ảnh tĩnh
3.2.1. Dữ liệu ảnh đầu vào
Dữ liệu ảnh được nhóm thu thập từ nhiều nguồn khác nhau như: ảnh thực tế chụp tại bãi đỗ xe, ảnh từ camera giám sát, ảnh từ internet… Các ảnh có sự đa dạng về điều kiện ánh sáng, góc chụp, loại biển số (biển trắng, biển xanh), chất lượng ảnh (rõ nét, mờ, nhiễu) nhằm mô phỏng sát điều kiện thực tế tại Việt Nam.
Thư mục ảnh đầu vào: https://drive.google.com/drive/folders/10y-oBJT3AGoUEoRWl2bCraDUXqPr1zPh?usp=drive_link
3.2.2. Quy trình xử lý
Đọc ảnh đầu vào: Sử dụng OpenCV để nạp ảnh từ thư mục hoặc đường dẫn URL, hiển thị ảnh để kiểm tra trực quan.
Nhận diện biển số bằng YOLOv8:
Huấn luyện mô hình YOLOv8 (fine-tuned với tập dữ liệu biển số Việt Nam).
        
               hình 3.2.2.1-phần kết quả train YOLOv8 cho các ảnh đầu vào
      Đoạn mã thể hiện biểu đồ sau khi huấn luyện YOLOv8:


           hình 3.2.2.2-Biểu đồ biến thiên loss trong quá trình huấn luyện mô hình YOLOv8
Dự đoán bounding box quanh vùng chứa biển số.
      
                       hình 3.2.2.3-bounding box vùng quanh biển số xe
Trả về tọa độ vùng biển số để crop ảnh.

                              hình 3.2.2.4-tọa độ sau khi crop
Tiền xử lý ảnh biển số:
Chuyển ảnh về grayscale.
Tăng tương phản và làm mịn ảnh.
Áp dụng nhị phân hóa (adaptive thresholding) để làm nổi bật ký tự.
   
        hình 3.2.2.5-Các bước tiền xử lý ảnh trước khi nhận dạng ký tự: (a) ảnh gốc, (b) ảnh grayscale, (c) tăng tương phản, (d) làm mịn, (e) nhị phân hóa
Nhận dạng ký tự (OCR):
EasyOCR: sử dụng mô hình CRNN, đọc chuỗi ký tự với độ tin cậy.
                
                            hình 3.2.2.6-sơ đồ quy trình nhận diện bằng easyocr
Tesseract: đọc ký tự bằng phương pháp LSTM, hoạt động hiệu quả khi ảnh rõ.
                      
                                hình 3.2.2.7-sơ đồ quy trình nhận diện bằng tesseract
Hiển thị kết quả:
Overlay kết quả lên ảnh gốc.
Ghi lại thời gian xử lý và độ chính xác (nếu có ground truth).
                       
                             hình 3.2.2.8-hình ảnh sau khi chạy chương trình 

3.2.3 Kết quả thử nghiệm

Tên ảnh
Kết quả EasyOCR
Kết quả Tesseract
Nhận xét
ảnh thẳng-sáng
nhận diện rõ ràng, chính xác
nhận diện rõ ràng, chính xác
Cả 2 đều nhận diện đúng
ảnh 
góc nghiêng - rõ
sai một kí tự khi nhận diện
nhận diện rõ ràng, chính xác
Tesseract nhận diện tốt hơn
ảnh 
mờ - thẳng
nhận diện rõ ràng, chính xác
chỉ nhận diện được các kí tự chữ
EasyOCR đọc nhanh hơn
ảnh nghiêng - mờ
sai một kí tự, nhận diện trong ảnh 
chỉ nhận diện ra được 2 con số, còn lại không nhận diện được
EasyOCR sai một kí tự, còn tesseract gần như không nhận diện được

                              bảng 3- bảng đánh giá nhận diện qua hình ảnh

EasyOCR cho kết quả ổn định hơn, đặc biệt với ảnh có chất lượng trung bình hoặc bị nhiễu. Tesseract đôi khi bỏ sót hoặc sai lệch do nhạy cảm với nhiễu ảnh và độ tương phản thấp.
3.3 Thử nghiệm với video
3.3.1. Dữ liệu video
Video được quay tại một số tuyến phố và bãi giữ xe trong khuôn viên trường đại học. Video có thời lượng từ 20 đến 40 giây, định dạng .mp4 hoặc .mov, khung hình từ 24–30 fps, độ phân giải 720p hoặc 1080p. Trong video có nhiều phương tiện với biển số chuyển động, góc quay nghiêng, ánh sáng thay đổi liên tục.
Đường dẫn cho video
3.3.2. Quy trình xử lý video
Nhận diện biển số trong video bằng OpenCV.
Duyệt từng khung hình (frame).
Thực hiện các bước như ảnh tĩnh: detect → crop → tiền xử lý → OCR.(bước này không nhất thiết vì nếu chỉ cần nhận diện biển số không thì không cần)
Overlay kết quả lên frame, hiển thị liên tục trong video output.
Ghi log kết quả OCR theo thời gian để đánh giá độ ổn định.
3.3.3. Kết quả quan sát

Video
EasyOCR
Tesseract
Nhận xét
video chứa ít biển số
nhận diện đúng
chỉ nhận diện được kí tự chữ
EasyOCR nhận diện ra được
video quay lại khi đi đường
không nhận diện được
không nhận diện được
cả 2 cái đều không nhận diện được ra biển số
video quay ở trong bãi đỗ xe
nhận diện ra được các biển số nhưng còn sai tương đối
chỉ nhận diện ra được kí tự chữ
cả 2 đều nhận diện ra được nhưng còn sai nhiều

                                           
                 bảng 4- Đánh giá kết quả và nhận xét khi nhận diện qua video

Trong môi trường video, EasyOCR chứng tỏ tính ổn định và khả năng chống nhiễu tốt hơn. Tesseract có tốc độ nhanh hơn nhưng dễ bỏ qua frame có ảnh bị mờ hoặc biến dạng do chuyển động.
3.4 Phân tích và đánh giá tổng quan
Nhóm đã tiến hành thử nghiệm trên hơn 30 ảnh và 6 video khác nhau, tổng cộng xử lý hơn 100 khung hình. Các tiêu chí được đánh giá bao gồm:
Tốc độ xử lí ảnh.
Độ chính xác nhận dạng ký tự
Khả năng thích ứng với điều kiện thực tế (ánh sáng yếu, góc nghiêng, ảnh rung)
Các đánh giá trên được rút ra từ quá trình thử nghiệm thực tế với ảnh và video trong nhiều điều kiện môi trường khác nhau.
Các hình ảnh minh họa cụ thể sẽ được trình bày tại Mục 3.5 – Minh họa kết quả.

Tiêu chí
EasyOCR
Tesseract
Độ chính xác
90-95%
50-70%
Tốc độ xử lí ảnh
~0.8-1.0s/ảnh
~0.5-0.7s/ảnh
Khả năng xử lí video
hay thiếu kí tự
hay thiếu sót kí tự hoặc mờ thì gần như không nhận diện được
Nhạy cảm với nhiễu
thấp
thấp
Dễ triển khai 
dễ
dễ
Hỗ trợ tiếng việt
cao
cần thêm cấu hình

                                 Bảng 5- phân tích và đánh giá tổng quan
3.5 Minh họa kết quả 
Hình 3.5.1: Ảnh đầu vào với bounding box biển số do YOLOv8 phát hiện. 
   
                                  

Ảnh đầu vào với vùng biển số được phát hiện bởi mô hình YOLOv8. Vùng bounding box màu đỏ biểu thị vị trí biển số được trích xuất phục vụ nhận dạng.
Hình 3.5.2: Biển số sau khi crop và tiền xử lý.
                           
                                Hình 3.5.2.1: Ảnh biển số sau khi crop


                                    Hình 3.5.2.2: Tiền xử lí hình ảnh

Crop biển số để YOLO nhận diện đúng vùng mình muốn nhận diện.
Tiền xử lí hình ảnh giúp nhận diện biển số trở nên dễ dàng hơn.

Hình 3.5.3: Kết quả nhận dạng ký tự bằng EasyOCR và Tesseract. 
    
                          
         Hình 3.5.3.1 – Ảnh biển số sau khi crop và kết quả EasyOCR 

 
                        
                Hình 3.5.3.2 – Ảnh biển số sau khi crop và kết quả Tesseract

Kết quả nhận dạng chuỗi ký tự từ ảnh đã xử lý thông qua nhận diện kí tự EasyOCR và Tesseract, so sánh kết quả giữa EasyOCR và Tesseract trên bảng số 4. EasyOCR cho kết quả gần đúng hơn trong ví dụ này.


Hình 3.5.4: Khung hình từ video có overlay kết quả OCR.



hình 3.5.4.1: kết quả nhận diện biển số trong video bằng EasyOCR


hình 3.5.4.2: kết quả nhận diện biển số trong video bằng Tesseract

Một khung hình từ video thực tế, hiển thị overlay vùng biển số và chuỗi ký tự đã nhận diện được.

3.6 Kết luận 
Qua quá trình thực nghiệm, nhóm nhận thấy hệ thống hoạt động ổn định, có thể áp dụng vào các tình huống thực tế như giám sát bãi xe, kiểm tra phương tiện tại cổng trường, khu dân cư. Mô hình YOLOv8 cho kết quả phát hiện biển số chính xác trong đa số trường hợp, còn EasyOCR có ưu thế rõ rệt trong việc nhận dạng ký tự trong điều kiện ảnh không lý tưởng. Tesseract tuy nhẹ và nhanh, nhưng đòi hỏi ảnh phải được xử lý kỹ, ít phù hợp với dữ liệu thực tế có nhiều biến động.
Từ đó, nhóm đề xuất các cải tiến sau:
Nâng cấp bộ dữ liệu với ảnh biển số đêm, ảnh từ camera hành trình.
Kết hợp thêm module kiểm tra lỗi và sửa lỗi ký tự dựa vào định dạng biển số Việt Nam.
Áp dụng mô hình nhận dạng mạnh hơn như TrOCR hoặc VietOCR để thay thế OCR truyền thống.


CHƯƠNG 4: KẾT LUẬN VÀ ĐỊNH HƯỚNG PHÁT TRIỂN
4.1 Những kết quả đạt được
Qua quá trình thực hiện đề tài "Xây dựng hệ thống nhận diện biển số xe tự động sử dụng YOLOv8 kết hợp EasyOCR và Tesseract", nhóm đã hoàn thành các mục tiêu đề ra ban đầu với những kết quả nổi bật sau:
Xây dựng thành công pipeline xử lý ảnh và video cho bài toán nhận diện biển số xe.
Sử dụng mô hình phát hiện đối tượng YOLOv8 để xác định chính xác vùng biển số trong ảnh đầu vào.
Kết hợp hai phương pháp OCR là EasyOCR và Tesseract để nhận diện ký tự trong biển số.
Triển khai hệ thống trên cả ảnh tĩnh và video, đáp ứng yêu cầu thời gian thực ở mức cơ bản.
Đánh giá, so sánh hiệu quả hai phương pháp OCR thông qua các tiêu chí độ chính xác, tốc độ và khả năng chịu nhiễu.
Hệ thống được xây dựng hoàn toàn bằng mã nguồn mở, dễ triển khai và có khả năng mở rộng cho các ứng dụng thực tế.
4.2 Hạn chế của đề tài
Bên cạnh những kết quả đạt được, hệ thống vẫn tồn tại một số điểm hạn chế cần được cải thiện:
Mô hình YOLOv8 chưa được huấn luyện chuyên sâu trên tập dữ liệu biển số Việt Nam đa dạng nên đôi khi phát hiện sai hoặc bỏ sót.
Tesseract OCR phụ thuộc mạnh vào chất lượng ảnh, khó nhận diện nếu ảnh bị mờ hoặc nghiêng.
Hệ thống chưa tích hợp cơ chế theo dõi nhiều khung hình (multi-frame tracking) để cải thiện độ chính xác trên video.
Giao diện người dùng còn đơn giản, chưa thân thiện với người không chuyên.
Chưa có cơ chế xử lý lỗi hoặc phân loại theo vùng miền, màu biển số.
4.3 Định hướng phát triển
Trong tương lai, nhóm mong muốn mở rộng và cải tiến hệ thống theo các hướng sau:
Nâng cấp tập dữ liệu: Thu thập thêm dữ liệu biển số Việt Nam trong nhiều điều kiện khác nhau để huấn luyện lại YOLOv8, tăng độ chính xác nhận diện.
Tối ưu hóa hiệu suất: Rút gọn mô hình để chạy hiệu quả hơn trên thiết bị nhúng (Raspberry Pi, Jetson Nano...).
Tích hợp giao diện đồ họa: Xây dựng phần mềm có giao diện GUI trực quan để dễ triển khai tại các bãi đỗ xe, khu dân cư.
Bổ sung phân loại biển số: Phân loại biển theo loại xe (xe công, xe tư nhân, nước ngoài...), vùng miền hoặc mục đích sử dụng.
Ứng dụng vào thực tế: Tích hợp hệ thống vào hệ thống giám sát giao thông, kiểm soát vào/ra tại cổng bảo vệ hoặc trạm thu phí.
4.4 Tổng kết
Đề tài đã giúp nhóm sinh viên áp dụng kiến thức về thị giác máy tính, học sâu và xử lý ảnh vào một bài toán thực tế có tính ứng dụng cao. Qua quá trình nghiên cứu, triển khai và đánh giá, nhóm đã tích lũy thêm nhiều kinh nghiệm về xử lý dữ liệu ảnh thực, sử dụng mô hình AI và làm việc nhóm hiệu quả.
Tuy còn tồn tại một số hạn chế, nhưng hệ thống nhận diện biển số được xây dựng đã chứng minh được tính khả thi, hiệu quả và tiềm năng mở rộng trong các bài toán thực tế.
