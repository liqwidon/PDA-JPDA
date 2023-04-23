#ifndef MARK_H
#define MARK_H

#include <array>
#include <Eigen/Dense>

typedef std::array< double, 3 > VectorPositionArray;
typedef std::array< double, 3 > VectorVelocityArray;
typedef std::array< double, 3 > VectorAccelerationArray;
typedef std::array< std::array< double, 3 >, 3 > Matrix3D;

class Mark
{
    bool m_isAssigned;

    double m_id;                        /// Идентификатор сообщения
    double m_rlsId;                     /// Идентификатор источника сообщения
    double m_trackId;                   /// Идентификатор частной трассы
    double m_time;                      /// Время измерения

    Eigen::Matrix<double, 3, 1> m_positionVector;
    Eigen::Matrix<double, 3, 1> m_velocityVector;
    Eigen::Matrix<double, 3, 1> m_accelerationVector;

    Eigen::Matrix<double, 3, 3> m_covPosPos;
    Eigen::Matrix<double, 3, 3> m_covVelVel;
    Eigen::Matrix<double, 3, 3> m_covAccAcc;
    Eigen::Matrix<double, 3, 3> m_covPosVel;
    Eigen::Matrix<double, 3, 3> m_covVelAcc;
    Eigen::Matrix<double, 3, 3> m_covPosAcc;
public:
    Mark(double id, double rlsid, double trackid)
        : m_id(id)
        , m_rlsId(rlsid)
        , m_trackId(trackid)
        , m_time(0)
    {}    ///копирование
    //    Mark(const Mark& another);

    virtual ~Mark() = default;

    Eigen::Matrix<double, 3, 1> getPositioin() const {
        return m_positionVector;
    }
    Eigen::Matrix<double, 3, 1> getVelocity() const;
    Eigen::Matrix<double, 3, 1> getAcceleration() const;
    double getTime() const;
    double getId() const;
    double getRlsId() const;
    double getTrackId() const;
    Eigen::Matrix<double, 3, 3> getcovPosPos() const {
        return m_covPosPos;
    }
    Eigen::Matrix<double, 3, 3> getDispPos() const;
    Eigen::Matrix<double, 3, 3> getDispVel() const;
    Eigen::Matrix<double, 3, 3> getDispAcc() const;
    Eigen::Matrix<double, 3, 3> getCovPosVel() const;
    Eigen::Matrix<double, 3, 3> getCovVelAcc() const;
    Eigen::Matrix<double, 3, 3> getCovPosAcc() const;




    void setPosition(const Eigen::Matrix<double,3,1> &vectorPos);
    void setPosition(const VectorPositionArray &arrPos);

    void setVelocity(const Eigen::Matrix<double,3,1> &vectorVel);
    void setVelocity(const VectorVelocityArray &arrVel);

    void setAcceleration(const Eigen::Matrix<double,3,1>& vectorAcc);
    void setAcceleration(const VectorAccelerationArray &arrAcc);

    void setTime(const double &time);
    void setIsAssigned(const bool &isAssigned);
    void setId(const double &id);
    void setTrackId(const double& tId);
    void setRlsId(const double& rlsId);

    void setDispPos(const Eigen::Matrix<double,3,3> &dispPos);
    void setDispVel(const Eigen::Matrix<double,3,3> &dispVel);
    void setDispAcc(const Eigen::Matrix<double,3,3> &dispAcc);
    void setCovPosVel(const Eigen::Matrix<double,3,3> &covPosVel);
    void setCovVelAcc(const Eigen::Matrix<double,3,3> &covVelAcc);
    void setCovPosAcc(const Eigen::Matrix<double,3,3> &covPosAcc);
    void setCovariances(const Eigen::Matrix<double,3,3> &dispPos, const Eigen::Matrix<double,3,3,0,3,3> &dispVel
                        , const Eigen::Matrix<double,3,3> &dispAcc, const Eigen::Matrix<double,3,3,0,3,3> &covPosVel
                        , const Eigen::Matrix<double,3,3> &covVelAcc, const Eigen::Matrix<double,3,3,0,3,3> &covPosAcc);

};

#endif // MARK_H
