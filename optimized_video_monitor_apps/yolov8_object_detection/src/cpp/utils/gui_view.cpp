#include "gui_view.h"

#define LabelDefaultStyle (          \
    "QLabel {"                       \
    "    font-size: 20px;"           \
    "    font-weight: bold;"         \
    "    color: #14FF39;"            \
    "    background-color: #646464;" \
    "    border: 2px solid black;"   \
    "    border-radius: 5px;"        \
    "    padding: 5px;"              \
    "    text-align: center;"        \
    "}")

#define PushButtonEnableStyle (      \
    "QPushButton {"                  \
    "    font-size: 19px;"           \
    "    font-weight: bold;"         \
    "    color: #14FF39;"            \
    "    background-color: #646464;" \
    "    padding: 5px;"              \
    "    text-align: center;"        \
    "}")

#define PushButtonDisableStyle (     \
    "QPushButton {"                  \
    "    font-size: 19px;"           \
    "    font-weight: bold;"         \
    "    color: #646464;"            \
    "    background-color: #646464;" \
    "    padding: 5px;"              \
    "    text-align: center;"        \
    "}")

constexpr int FRAME_BUFFER_SIZE = 60;
constexpr float CONFIDENCE_INCREMENT = 0.05f;
constexpr float MAX_CONFIDENCE = 0.95f;
constexpr float MIN_CONFIDENCE = 0.1f;
std::mutex inf_type_mutex;

YOLOGuiView::YOLOGuiView(int &argc, char *argv[]) : app(argc, argv)
{
    int width_offset = 0;

    int num_screens = app.screens().size();

    if (num_screens < 1)
    {
        printf("No display screen detected\n");
        QCoreApplication::exit(1);
    }

    for (int i = 0; i < num_screens; i++)
    {
        QScreen *qscreen = app.screens().at(i);
        auto screen = std::make_unique<DisplayScreen>(nullptr, qscreen);
        int w = qscreen->geometry().width();
        int h = qscreen->geometry().height();
        screen->setGeometry(width_offset, 0, w, h);
        width_offset += w;
        screens.push_back(std::move(screen));
    }
}

int YOLOGuiView::Run()
{
    return app.exec();
}

void YOLOGuiView::Exit()
{
    QCoreApplication::exit(0);
}

YOLOGuiView::~YOLOGuiView()
{
    screens.clear();
}

DisplayScreen::DisplayScreen() : QWidget() // default constructor
{
    this->running_ = true;
}

DisplayScreen::DisplayScreen(QWidget *parent = nullptr, QScreen *qscreen = nullptr) : QWidget(parent)
{
    this->running_ = true;
    if (!qscreen)
    {
        throw std::invalid_argument("QScreen cannot be null");
    }

    this->w_ = qscreen->geometry().width();
    this->h_ = qscreen->geometry().height();
}

DisplayScreen::~DisplayScreen()
{
    this->running_ = false;
    for (uint32_t idx = 0; idx < this->num_viewers_; idx++)
    {
        FrameViewer *viewer = viewers_[idx];
        if (viewer)
            delete viewer;
    }
}

void DisplayScreen::Show()
{
    this->show();
}

// TODO: make it a function directly in FrameViewer
void DisplayScreen::SetDisplayFrame(int viewer_id, cv::Mat *frame, float fps)
{
    FrameViewer *viewer = viewers_[viewer_id];
    viewer->UpdateFrame(frame);
    viewer->UpdateFPS(fps);
}

void DisplayScreen::SetDisplayFrame(int viewer_id, cv::Mat *frame)
{
    FrameViewer *viewer = viewers_[viewer_id];
    viewer->UpdateFrame(frame);
    viewer->HideFPS();
    viewer->HideChannelName();
}

void DisplayScreen::SetDisplayFrame(int viewer_id, cv::Mat frame)
{
    SetDisplayFrame(viewer_id, &frame);
}

cv::Mat *DisplayScreen::GetDisplayFrameBuf(int viewer_id)
{
    FrameViewer *viewer = viewers_[viewer_id];
    cv::Mat *display_frame = viewer->display_frame_list_.at(viewer->display_frame_idx_);
    viewer->display_frame_idx_ = (viewer->display_frame_idx_ + 1) % viewer->display_frame_list_.size();
    return display_frame;
}

uint32_t DisplayScreen::width()
{
    return this->w_;
}

uint32_t DisplayScreen::height()
{
    return this->h_;
}

uint32_t DisplayScreen::GetViewerWidth(int viewer_id)
{
    FrameViewer *viewer = viewers_[viewer_id];
    return viewer->width();
}

uint32_t DisplayScreen::GetViewerHeight(int viewer_id)
{
    FrameViewer *viewer = viewers_[viewer_id];
    return viewer->height();
}

void DisplayScreen::SetSquareLayout(int num_channels)
{
    static int idx = 0;
    static bool create_exit_button = false;
    int screen_width = this->width();
    int screen_height = this->height();

    std::vector<ViewerGeometry> viewer_geometry;
    printf("screen resolution = %dx%d\n", screen_width, screen_height);

    {
        // NxN layout
        int mode = ceil(sqrt(num_channels));

        int geo_w, geo_h;
        geo_w = int((screen_width) / mode);
        geo_w &= (~0x1f); // sws_scale needs width to be 32x
        geo_h = int((screen_height) / mode);
        geo_h = (geo_w * 9) / 16;

        viewer_geometry.clear();

        for (int i = 0; i < num_channels; i++)
        {
            int x = (i % mode) * geo_w;
            int y = ((i / mode) * (geo_h));
            ViewerGeometry cg = {x, y, geo_w, geo_h};
            viewer_geometry.push_back(cg);
        }

        this->viewer_geometry_ = viewer_geometry;
    }

    // init and set all necessary components in each widget

    for (int i = 0; i < num_channels; i++)
    {
        ViewerGeometry &cg = viewer_geometry.at(i);

        FrameViewer *viewer = new FrameViewer(this, true);
        viewer->SetGeometry(cg.x, cg.y, cg.w, cg.h);
        viewer->SetIdx(idx++); // set id for parent to distinguish
        viewer->HideConfidence();
        viewer->HideModelName();
        this->AddViewer(viewer);
    }

    if (!create_exit_button)
    {
        QPushButton *exitButton = new QPushButton("Exit", this);
        exitButton->setGeometry(screen_width - 60, 0, 60, 25); // FIXME
        QObject::connect(exitButton, &QPushButton::clicked, &QCoreApplication::quit);
        create_exit_button = true;
    }

    this->setWindowState(Qt::WindowFullScreen);
}

void DisplayScreen::AddViewer(FrameViewer *viewer)
{
    viewers_.push_back(viewer);
    num_viewers_ = viewers_.size();
}

void DisplayScreen::SetConfidenceValue(float confidence)
{
    for (uint32_t idx = 0; idx < this->num_viewers_; idx++)
    {
        FrameViewer *viewer = viewers_[idx];
        viewer->confidence_ = confidence;
    }
    viewers_[0]->UpdateConfidence(); /*TODO*/
}

float DisplayScreen::GetConfidenceValue()
{
    return viewers_[0]->confidence_; /*TODO*/
}

void DisplayScreen::SetModelName(std::string model_name)
{
    viewers_[0]->UpdateModelName(model_name.c_str()); /*TODO*/
}

uint32_t DisplayScreen::NumViewers()
{
    return num_viewers_;
}

FrameViewer::FrameViewer(QWidget *parent = nullptr, bool show_confidence = true) : QWidget(parent)
{
    connect(this, SIGNAL(SignalUpdateFrame(cv::Mat *)), this, SLOT(SlotUpdateFrame(cv::Mat *)));
    connect(this, SIGNAL(SignalUpdateFPS(float)), this, SLOT(SlotUpdateFPS(float)));
    connect(this, SIGNAL(SignalUpdateConfidence()), this, SLOT(SlotUpdateConfidence()));

    this->running_ = true;
    this->inf_type_ = NONE_INFERENCE;
    this->display_frame_idx_ = 0;
    this->confidence_ = -1.0;

    frame_ = new QLabel(this);

    int count = 0;
    int interval = 35; // TODO

    name_ = new QLabel(frame_);
    name_->setStyleSheet(LabelDefaultStyle);
    name_->move(0, count * interval);

    fps_ = new QLabel("FPS = ", frame_);
    fps_->setStyleSheet(LabelDefaultStyle);
    fps_->move(name_->width() - 3, count * interval);
    count++;

    if (show_confidence)
    {
        confidence_label_ = new QLabel(frame_);
        confidence_label_->setStyleSheet(LabelDefaultStyle);
        confidence_label_->move(0, count * interval);
        confidence_label_->setText("confidence = " + QString::number(0.30, 'f', 2));
        confidence_label_->adjustSize();

        /* Condidence Buttons*/
        confidence_button_widget = new QWidget(this);
        QFont demo_font("Montserrat");
        confidence_button_widget->setFont(demo_font);
        confidence_button_widget->setProperty("displayName", "My Display Name");
        confidence_button_widget->setGeometry(confidence_label_->width(), count * interval, 50, confidence_label_->height() * 1.1);

        button_up = new QPushButton(confidence_button_widget);
        button_down = new QPushButton(confidence_button_widget);

        button_up->setProperty("displayName", "+0.1");
        button_up->setToolTip(button_up->property("displayName").toString());
        button_down->setProperty("displayName", "-0.1");
        button_down->setToolTip(button_down->property("displayName").toString());
        QCommonStyle style;
        button_up->setIcon(style.standardIcon(QStyle::SP_ArrowUp));
        button_down->setIcon(style.standardIcon(QStyle::SP_ArrowDown));

        QObject::connect(button_up, &QPushButton::clicked, this, &FrameViewer::SlotConfAdd);
        QObject::connect(button_down, &QPushButton::clicked, this, &FrameViewer::SlotConfReduce);

        QVBoxLayout *vertical_layout_right = new QVBoxLayout();
        vertical_layout_right->addWidget(button_up);
        vertical_layout_right->addWidget(button_down);
        confidence_button_widget->setLayout(vertical_layout_right);

        count++;
    }
    model_label_ = new QLabel(frame_);
    model_label_->setStyleSheet(LabelDefaultStyle);
    model_label_->move(0, count * interval);
    count++;

    // Set up the layout
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(frame_);
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);
}

FrameViewer::~FrameViewer()
{
    this->running_ = false;
    int size = display_frame_list_.size();
    for (int i = 0; i < size; i++)
        if (display_frame_list_[i])
            delete display_frame_list_[i];
}

uint32_t FrameViewer::width()
{
    return this->w_;
}

uint32_t FrameViewer::height()
{
    return this->h_;
}

void FrameViewer::SetGeometry(int x, int y, int w, int h)
{
    x_ = x;
    y_ = y;
    w_ = w;
    h_ = h;
    this->setGeometry(x, y, w, h);
    // Set up display frame buffer
    for (int i = 0; i < FRAME_BUFFER_SIZE /* TODO */; i++)
    {
        cv::Mat *display_frame = new cv::Mat(h_, w_, CV_8UC3);
        this->display_frame_list_.push_back(display_frame);
    }
}

void FrameViewer::SetIdx(int idx)
{
    this->idx_ = idx;
    this->setObjectName(QString::number(idx_ + 1));
    name_->setText("CH" + QString::number(idx_ + 1)); // one-index
    name_->adjustSize();
}

void FrameViewer::UpdateFrame(cv::Mat *frame)
{
    emit SignalUpdateFrame(frame);
}

void FrameViewer::SlotUpdateFrame(cv::Mat *frame)
{
    QImage img((*frame).data, (*frame).cols, (*frame).rows, (*frame).step, QImage::Format_RGB888);
    // Set the QImage as the pixmap for the QLabel
    frame_->setPixmap(QPixmap::fromImage(img));
}

void FrameViewer::UpdateFPS(float fps)
{
    emit SignalUpdateFPS(fps);
}

void FrameViewer::SlotUpdateFPS(float fps)
{
    if (fps == .0)
        return;
    fps_->move(name_->width() - 3, 0);
    fps_->setText("FPS = " + QString::number(fps, 'f', 1));
    fps_->adjustSize();
}

void FrameViewer::UpdateConfidence()
{
    emit SignalUpdateConfidence();
}

void FrameViewer::SlotUpdateConfidence()
{
    confidence_label_->show();
    confidence_button_widget->show();
    confidence_label_->setText("confidence = " + QString::number(confidence_, 'f', 2));
    confidence_label_->adjustSize();
}

void FrameViewer::HideConfidence()
{
    confidence_label_->hide();
    confidence_button_widget->hide();
}

void FrameViewer::HideFPS()
{
    this->fps_->hide();
}

void FrameViewer::HideChannelName()
{
    this->name_->hide();
}

void FrameViewer::HideModelName()
{
    this->model_label_->hide();
}

void FrameViewer::UpdateModelName(const char *model_name)
{
    model_label_->show();
    model_label_->setText(model_name);
    model_label_->adjustSize();
}

void FrameViewer::SlotConfAdd()
{
    if (confidence_ >= MAX_CONFIDENCE)
        return;

    confidence_ += CONFIDENCE_INCREMENT;
    UpdateConfidence();
}

void FrameViewer::SlotConfReduce()
{
    if (confidence_ <= MIN_CONFIDENCE)
        return;

    confidence_ -= CONFIDENCE_INCREMENT;
    UpdateConfidence();
}
