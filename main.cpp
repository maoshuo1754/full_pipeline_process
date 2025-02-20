//
// Created by csic724 on 2024/11/11.
//
#include "Config.h"
#include "ThreadPool.h"
#include "DataSource.h"

int main(){
    // config read and monitor thread
    string configFilePath = "../config.json";
    loadConfig(configFilePath);
    std::thread configMonitor(monitorConfig, configFilePath, loadConfig);

    // data source thread
    unique_ptr<DataSource> dataSource;
    if(dataSource_type == 0) {
        dataSource = make_unique<FileDataSource>(dataPath, monitorWriterRunning);
    } else {
        dataSource = make_unique<XDMADataSource>(monitorWriterRunning);
    }
    std::thread dataSourceThread(&DataSource::run, dataSource.get());

    // thread_pool
    auto* sharedQueue = initSharedMemery(true);
    cout << "Working Threads Num:" << num_threads << endl;
    ThreadPool thread_pool(num_threads, sharedQueue);
    thread_pool.run();

    // wait threads end
    monitorConfigRunning = false;
    monitorWriterRunning = false;
    configMonitor.join();
    dataSourceThread.join();
}
