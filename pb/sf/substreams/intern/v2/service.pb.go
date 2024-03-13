// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        (unknown)
// source: sf/substreams/intern/v2/service.proto

package pbssinternal

import (
	v1 "github.com/streamingfast/substreams/pb/sf/substreams/v1"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	_ "google.golang.org/protobuf/types/known/anypb"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type ProcessRangeRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	StartBlockNum uint64      `protobuf:"varint,1,opt,name=start_block_num,json=startBlockNum,proto3" json:"start_block_num,omitempty"`
	StopBlockNum  uint64      `protobuf:"varint,2,opt,name=stop_block_num,json=stopBlockNum,proto3" json:"stop_block_num,omitempty"`
	OutputModule  string      `protobuf:"bytes,3,opt,name=output_module,json=outputModule,proto3" json:"output_module,omitempty"`
	Modules       *v1.Modules `protobuf:"bytes,4,opt,name=modules,proto3" json:"modules,omitempty"`
	Stage         uint32      `protobuf:"varint,5,opt,name=stage,proto3" json:"stage,omitempty"` // 0-based index of stage to execute up to
}

func (x *ProcessRangeRequest) Reset() {
	*x = ProcessRangeRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessRangeRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessRangeRequest) ProtoMessage() {}

func (x *ProcessRangeRequest) ProtoReflect() protoreflect.Message {
	mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessRangeRequest.ProtoReflect.Descriptor instead.
func (*ProcessRangeRequest) Descriptor() ([]byte, []int) {
	return file_sf_substreams_intern_v2_service_proto_rawDescGZIP(), []int{0}
}

func (x *ProcessRangeRequest) GetStartBlockNum() uint64 {
	if x != nil {
		return x.StartBlockNum
	}
	return 0
}

func (x *ProcessRangeRequest) GetStopBlockNum() uint64 {
	if x != nil {
		return x.StopBlockNum
	}
	return 0
}

func (x *ProcessRangeRequest) GetOutputModule() string {
	if x != nil {
		return x.OutputModule
	}
	return ""
}

func (x *ProcessRangeRequest) GetModules() *v1.Modules {
	if x != nil {
		return x.Modules
	}
	return nil
}

func (x *ProcessRangeRequest) GetStage() uint32 {
	if x != nil {
		return x.Stage
	}
	return 0
}

type ProcessRangeResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Types that are assignable to Type:
	//
	//	*ProcessRangeResponse_Failed
	//	*ProcessRangeResponse_Completed
	//	*ProcessRangeResponse_Update
	Type isProcessRangeResponse_Type `protobuf_oneof:"type"`
}

func (x *ProcessRangeResponse) Reset() {
	*x = ProcessRangeResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ProcessRangeResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ProcessRangeResponse) ProtoMessage() {}

func (x *ProcessRangeResponse) ProtoReflect() protoreflect.Message {
	mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ProcessRangeResponse.ProtoReflect.Descriptor instead.
func (*ProcessRangeResponse) Descriptor() ([]byte, []int) {
	return file_sf_substreams_intern_v2_service_proto_rawDescGZIP(), []int{1}
}

func (m *ProcessRangeResponse) GetType() isProcessRangeResponse_Type {
	if m != nil {
		return m.Type
	}
	return nil
}

func (x *ProcessRangeResponse) GetFailed() *Failed {
	if x, ok := x.GetType().(*ProcessRangeResponse_Failed); ok {
		return x.Failed
	}
	return nil
}

func (x *ProcessRangeResponse) GetCompleted() *Completed {
	if x, ok := x.GetType().(*ProcessRangeResponse_Completed); ok {
		return x.Completed
	}
	return nil
}

func (x *ProcessRangeResponse) GetUpdate() *Update {
	if x, ok := x.GetType().(*ProcessRangeResponse_Update); ok {
		return x.Update
	}
	return nil
}

type isProcessRangeResponse_Type interface {
	isProcessRangeResponse_Type()
}

type ProcessRangeResponse_Failed struct {
	Failed *Failed `protobuf:"bytes,4,opt,name=failed,proto3,oneof"`
}

type ProcessRangeResponse_Completed struct {
	Completed *Completed `protobuf:"bytes,5,opt,name=completed,proto3,oneof"`
}

type ProcessRangeResponse_Update struct {
	Update *Update `protobuf:"bytes,6,opt,name=update,proto3,oneof"`
}

func (*ProcessRangeResponse_Failed) isProcessRangeResponse_Type() {}

func (*ProcessRangeResponse_Completed) isProcessRangeResponse_Type() {}

func (*ProcessRangeResponse_Update) isProcessRangeResponse_Type() {}

type Update struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	DurationMs        uint64         `protobuf:"varint,1,opt,name=duration_ms,json=durationMs,proto3" json:"duration_ms,omitempty"`
	ProcessedBlocks   uint64         `protobuf:"varint,2,opt,name=processed_blocks,json=processedBlocks,proto3" json:"processed_blocks,omitempty"`
	TotalBytesRead    uint64         `protobuf:"varint,3,opt,name=total_bytes_read,json=totalBytesRead,proto3" json:"total_bytes_read,omitempty"`
	TotalBytesWritten uint64         `protobuf:"varint,4,opt,name=total_bytes_written,json=totalBytesWritten,proto3" json:"total_bytes_written,omitempty"`
	ModulesStats      []*ModuleStats `protobuf:"bytes,5,rep,name=modules_stats,json=modulesStats,proto3" json:"modules_stats,omitempty"`
}

func (x *Update) Reset() {
	*x = Update{}
	if protoimpl.UnsafeEnabled {
		mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Update) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Update) ProtoMessage() {}

func (x *Update) ProtoReflect() protoreflect.Message {
	mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Update.ProtoReflect.Descriptor instead.
func (*Update) Descriptor() ([]byte, []int) {
	return file_sf_substreams_intern_v2_service_proto_rawDescGZIP(), []int{2}
}

func (x *Update) GetDurationMs() uint64 {
	if x != nil {
		return x.DurationMs
	}
	return 0
}

func (x *Update) GetProcessedBlocks() uint64 {
	if x != nil {
		return x.ProcessedBlocks
	}
	return 0
}

func (x *Update) GetTotalBytesRead() uint64 {
	if x != nil {
		return x.TotalBytesRead
	}
	return 0
}

func (x *Update) GetTotalBytesWritten() uint64 {
	if x != nil {
		return x.TotalBytesWritten
	}
	return 0
}

func (x *Update) GetModulesStats() []*ModuleStats {
	if x != nil {
		return x.ModulesStats
	}
	return nil
}

type ModuleStats struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Name                 string                `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	ProcessingTimeMs     uint64                `protobuf:"varint,2,opt,name=processing_time_ms,json=processingTimeMs,proto3" json:"processing_time_ms,omitempty"`
	StoreOperationTimeMs uint64                `protobuf:"varint,3,opt,name=store_operation_time_ms,json=storeOperationTimeMs,proto3" json:"store_operation_time_ms,omitempty"`
	StoreReadCount       uint64                `protobuf:"varint,4,opt,name=store_read_count,json=storeReadCount,proto3" json:"store_read_count,omitempty"`
	ExternalCallMetrics  []*ExternalCallMetric `protobuf:"bytes,5,rep,name=external_call_metrics,json=externalCallMetrics,proto3" json:"external_call_metrics,omitempty"`
	// store-specific (will be 0 on mappers)
	StoreWriteCount        uint64 `protobuf:"varint,10,opt,name=store_write_count,json=storeWriteCount,proto3" json:"store_write_count,omitempty"`
	StoreDeleteprefixCount uint64 `protobuf:"varint,11,opt,name=store_deleteprefix_count,json=storeDeleteprefixCount,proto3" json:"store_deleteprefix_count,omitempty"`
	StoreSizeBytes         uint64 `protobuf:"varint,12,opt,name=store_size_bytes,json=storeSizeBytes,proto3" json:"store_size_bytes,omitempty"`
}

func (x *ModuleStats) Reset() {
	*x = ModuleStats{}
	if protoimpl.UnsafeEnabled {
		mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ModuleStats) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ModuleStats) ProtoMessage() {}

func (x *ModuleStats) ProtoReflect() protoreflect.Message {
	mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ModuleStats.ProtoReflect.Descriptor instead.
func (*ModuleStats) Descriptor() ([]byte, []int) {
	return file_sf_substreams_intern_v2_service_proto_rawDescGZIP(), []int{3}
}

func (x *ModuleStats) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *ModuleStats) GetProcessingTimeMs() uint64 {
	if x != nil {
		return x.ProcessingTimeMs
	}
	return 0
}

func (x *ModuleStats) GetStoreOperationTimeMs() uint64 {
	if x != nil {
		return x.StoreOperationTimeMs
	}
	return 0
}

func (x *ModuleStats) GetStoreReadCount() uint64 {
	if x != nil {
		return x.StoreReadCount
	}
	return 0
}

func (x *ModuleStats) GetExternalCallMetrics() []*ExternalCallMetric {
	if x != nil {
		return x.ExternalCallMetrics
	}
	return nil
}

func (x *ModuleStats) GetStoreWriteCount() uint64 {
	if x != nil {
		return x.StoreWriteCount
	}
	return 0
}

func (x *ModuleStats) GetStoreDeleteprefixCount() uint64 {
	if x != nil {
		return x.StoreDeleteprefixCount
	}
	return 0
}

func (x *ModuleStats) GetStoreSizeBytes() uint64 {
	if x != nil {
		return x.StoreSizeBytes
	}
	return 0
}

type ExternalCallMetric struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Name   string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
	Count  uint64 `protobuf:"varint,2,opt,name=count,proto3" json:"count,omitempty"`
	TimeMs uint64 `protobuf:"varint,3,opt,name=time_ms,json=timeMs,proto3" json:"time_ms,omitempty"`
}

func (x *ExternalCallMetric) Reset() {
	*x = ExternalCallMetric{}
	if protoimpl.UnsafeEnabled {
		mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ExternalCallMetric) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ExternalCallMetric) ProtoMessage() {}

func (x *ExternalCallMetric) ProtoReflect() protoreflect.Message {
	mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ExternalCallMetric.ProtoReflect.Descriptor instead.
func (*ExternalCallMetric) Descriptor() ([]byte, []int) {
	return file_sf_substreams_intern_v2_service_proto_rawDescGZIP(), []int{4}
}

func (x *ExternalCallMetric) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

func (x *ExternalCallMetric) GetCount() uint64 {
	if x != nil {
		return x.Count
	}
	return 0
}

func (x *ExternalCallMetric) GetTimeMs() uint64 {
	if x != nil {
		return x.TimeMs
	}
	return 0
}

type Completed struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	AllProcessedRanges []*BlockRange `protobuf:"bytes,1,rep,name=all_processed_ranges,json=allProcessedRanges,proto3" json:"all_processed_ranges,omitempty"`
	// TraceId represents the producer's trace id that produced the partial files.
	// This is present here so that the consumer can use it to identify the
	// right partial files that needs to be squashed together.
	//
	// The TraceId can be empty in which case it should be assumed by the tier1
	// consuming this message that the tier2 that produced those partial files
	// is not yet updated to produce a trace id and a such, the tier1 should
	// generate a legacy partial file name.
	TraceId string `protobuf:"bytes,2,opt,name=trace_id,json=traceId,proto3" json:"trace_id,omitempty"`
}

func (x *Completed) Reset() {
	*x = Completed{}
	if protoimpl.UnsafeEnabled {
		mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Completed) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Completed) ProtoMessage() {}

func (x *Completed) ProtoReflect() protoreflect.Message {
	mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[5]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Completed.ProtoReflect.Descriptor instead.
func (*Completed) Descriptor() ([]byte, []int) {
	return file_sf_substreams_intern_v2_service_proto_rawDescGZIP(), []int{5}
}

func (x *Completed) GetAllProcessedRanges() []*BlockRange {
	if x != nil {
		return x.AllProcessedRanges
	}
	return nil
}

func (x *Completed) GetTraceId() string {
	if x != nil {
		return x.TraceId
	}
	return ""
}

type Failed struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Reason string   `protobuf:"bytes,1,opt,name=reason,proto3" json:"reason,omitempty"`
	Logs   []string `protobuf:"bytes,2,rep,name=logs,proto3" json:"logs,omitempty"`
	// FailureLogsTruncated is a flag that tells you if you received all the logs or if they
	// were truncated because you logged too much (fixed limit currently is set to 128 KiB).
	LogsTruncated bool `protobuf:"varint,3,opt,name=logs_truncated,json=logsTruncated,proto3" json:"logs_truncated,omitempty"`
}

func (x *Failed) Reset() {
	*x = Failed{}
	if protoimpl.UnsafeEnabled {
		mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Failed) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Failed) ProtoMessage() {}

func (x *Failed) ProtoReflect() protoreflect.Message {
	mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[6]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Failed.ProtoReflect.Descriptor instead.
func (*Failed) Descriptor() ([]byte, []int) {
	return file_sf_substreams_intern_v2_service_proto_rawDescGZIP(), []int{6}
}

func (x *Failed) GetReason() string {
	if x != nil {
		return x.Reason
	}
	return ""
}

func (x *Failed) GetLogs() []string {
	if x != nil {
		return x.Logs
	}
	return nil
}

func (x *Failed) GetLogsTruncated() bool {
	if x != nil {
		return x.LogsTruncated
	}
	return false
}

type BlockRange struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	StartBlock uint64 `protobuf:"varint,2,opt,name=start_block,json=startBlock,proto3" json:"start_block,omitempty"`
	EndBlock   uint64 `protobuf:"varint,3,opt,name=end_block,json=endBlock,proto3" json:"end_block,omitempty"`
}

func (x *BlockRange) Reset() {
	*x = BlockRange{}
	if protoimpl.UnsafeEnabled {
		mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *BlockRange) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*BlockRange) ProtoMessage() {}

func (x *BlockRange) ProtoReflect() protoreflect.Message {
	mi := &file_sf_substreams_intern_v2_service_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use BlockRange.ProtoReflect.Descriptor instead.
func (*BlockRange) Descriptor() ([]byte, []int) {
	return file_sf_substreams_intern_v2_service_proto_rawDescGZIP(), []int{7}
}

func (x *BlockRange) GetStartBlock() uint64 {
	if x != nil {
		return x.StartBlock
	}
	return 0
}

func (x *BlockRange) GetEndBlock() uint64 {
	if x != nil {
		return x.EndBlock
	}
	return 0
}

var File_sf_substreams_intern_v2_service_proto protoreflect.FileDescriptor

var file_sf_substreams_intern_v2_service_proto_rawDesc = []byte{
	0x0a, 0x25, 0x73, 0x66, 0x2f, 0x73, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2f,
	0x69, 0x6e, 0x74, 0x65, 0x72, 0x6e, 0x2f, 0x76, 0x32, 0x2f, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63,
	0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x19, 0x73, 0x66, 0x2e, 0x73, 0x75, 0x62, 0x73,
	0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e, 0x69, 0x6e, 0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0x2e,
	0x76, 0x32, 0x1a, 0x19, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x62, 0x75, 0x66, 0x2f, 0x61, 0x6e, 0x79, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x1e, 0x73,
	0x66, 0x2f, 0x73, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2f, 0x76, 0x31, 0x2f,
	0x6d, 0x6f, 0x64, 0x75, 0x6c, 0x65, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0xd3, 0x01,
	0x0a, 0x13, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x52, 0x61, 0x6e, 0x67, 0x65, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x26, 0x0a, 0x0f, 0x73, 0x74, 0x61, 0x72, 0x74, 0x5f, 0x62,
	0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x6e, 0x75, 0x6d, 0x18, 0x01, 0x20, 0x01, 0x28, 0x04, 0x52, 0x0d,
	0x73, 0x74, 0x61, 0x72, 0x74, 0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x4e, 0x75, 0x6d, 0x12, 0x24, 0x0a,
	0x0e, 0x73, 0x74, 0x6f, 0x70, 0x5f, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x5f, 0x6e, 0x75, 0x6d, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x04, 0x52, 0x0c, 0x73, 0x74, 0x6f, 0x70, 0x42, 0x6c, 0x6f, 0x63, 0x6b,
	0x4e, 0x75, 0x6d, 0x12, 0x23, 0x0a, 0x0d, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x6d, 0x6f,
	0x64, 0x75, 0x6c, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c, 0x6f, 0x75, 0x74, 0x70,
	0x75, 0x74, 0x4d, 0x6f, 0x64, 0x75, 0x6c, 0x65, 0x12, 0x33, 0x0a, 0x07, 0x6d, 0x6f, 0x64, 0x75,
	0x6c, 0x65, 0x73, 0x18, 0x04, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x19, 0x2e, 0x73, 0x66, 0x2e, 0x73,
	0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e, 0x76, 0x31, 0x2e, 0x4d, 0x6f, 0x64,
	0x75, 0x6c, 0x65, 0x73, 0x52, 0x07, 0x6d, 0x6f, 0x64, 0x75, 0x6c, 0x65, 0x73, 0x12, 0x14, 0x0a,
	0x05, 0x73, 0x74, 0x61, 0x67, 0x65, 0x18, 0x05, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x05, 0x73, 0x74,
	0x61, 0x67, 0x65, 0x22, 0xf0, 0x01, 0x0a, 0x14, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x52,
	0x61, 0x6e, 0x67, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x3b, 0x0a, 0x06,
	0x66, 0x61, 0x69, 0x6c, 0x65, 0x64, 0x18, 0x04, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x21, 0x2e, 0x73,
	0x66, 0x2e, 0x73, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e, 0x69, 0x6e, 0x74,
	0x65, 0x72, 0x6e, 0x61, 0x6c, 0x2e, 0x76, 0x32, 0x2e, 0x46, 0x61, 0x69, 0x6c, 0x65, 0x64, 0x48,
	0x00, 0x52, 0x06, 0x66, 0x61, 0x69, 0x6c, 0x65, 0x64, 0x12, 0x44, 0x0a, 0x09, 0x63, 0x6f, 0x6d,
	0x70, 0x6c, 0x65, 0x74, 0x65, 0x64, 0x18, 0x05, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x24, 0x2e, 0x73,
	0x66, 0x2e, 0x73, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e, 0x69, 0x6e, 0x74,
	0x65, 0x72, 0x6e, 0x61, 0x6c, 0x2e, 0x76, 0x32, 0x2e, 0x43, 0x6f, 0x6d, 0x70, 0x6c, 0x65, 0x74,
	0x65, 0x64, 0x48, 0x00, 0x52, 0x09, 0x63, 0x6f, 0x6d, 0x70, 0x6c, 0x65, 0x74, 0x65, 0x64, 0x12,
	0x3b, 0x0a, 0x06, 0x75, 0x70, 0x64, 0x61, 0x74, 0x65, 0x18, 0x06, 0x20, 0x01, 0x28, 0x0b, 0x32,
	0x21, 0x2e, 0x73, 0x66, 0x2e, 0x73, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e,
	0x69, 0x6e, 0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0x2e, 0x76, 0x32, 0x2e, 0x55, 0x70, 0x64, 0x61,
	0x74, 0x65, 0x48, 0x00, 0x52, 0x06, 0x75, 0x70, 0x64, 0x61, 0x74, 0x65, 0x42, 0x06, 0x0a, 0x04,
	0x74, 0x79, 0x70, 0x65, 0x4a, 0x04, 0x08, 0x01, 0x10, 0x02, 0x4a, 0x04, 0x08, 0x02, 0x10, 0x03,
	0x4a, 0x04, 0x08, 0x03, 0x10, 0x04, 0x22, 0xfb, 0x01, 0x0a, 0x06, 0x55, 0x70, 0x64, 0x61, 0x74,
	0x65, 0x12, 0x1f, 0x0a, 0x0b, 0x64, 0x75, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x6d, 0x73,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x04, 0x52, 0x0a, 0x64, 0x75, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e,
	0x4d, 0x73, 0x12, 0x29, 0x0a, 0x10, 0x70, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x65, 0x64, 0x5f,
	0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x04, 0x52, 0x0f, 0x70, 0x72,
	0x6f, 0x63, 0x65, 0x73, 0x73, 0x65, 0x64, 0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x73, 0x12, 0x28, 0x0a,
	0x10, 0x74, 0x6f, 0x74, 0x61, 0x6c, 0x5f, 0x62, 0x79, 0x74, 0x65, 0x73, 0x5f, 0x72, 0x65, 0x61,
	0x64, 0x18, 0x03, 0x20, 0x01, 0x28, 0x04, 0x52, 0x0e, 0x74, 0x6f, 0x74, 0x61, 0x6c, 0x42, 0x79,
	0x74, 0x65, 0x73, 0x52, 0x65, 0x61, 0x64, 0x12, 0x2e, 0x0a, 0x13, 0x74, 0x6f, 0x74, 0x61, 0x6c,
	0x5f, 0x62, 0x79, 0x74, 0x65, 0x73, 0x5f, 0x77, 0x72, 0x69, 0x74, 0x74, 0x65, 0x6e, 0x18, 0x04,
	0x20, 0x01, 0x28, 0x04, 0x52, 0x11, 0x74, 0x6f, 0x74, 0x61, 0x6c, 0x42, 0x79, 0x74, 0x65, 0x73,
	0x57, 0x72, 0x69, 0x74, 0x74, 0x65, 0x6e, 0x12, 0x4b, 0x0a, 0x0d, 0x6d, 0x6f, 0x64, 0x75, 0x6c,
	0x65, 0x73, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x73, 0x18, 0x05, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x26,
	0x2e, 0x73, 0x66, 0x2e, 0x73, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e, 0x69,
	0x6e, 0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0x2e, 0x76, 0x32, 0x2e, 0x4d, 0x6f, 0x64, 0x75, 0x6c,
	0x65, 0x53, 0x74, 0x61, 0x74, 0x73, 0x52, 0x0c, 0x6d, 0x6f, 0x64, 0x75, 0x6c, 0x65, 0x73, 0x53,
	0x74, 0x61, 0x74, 0x73, 0x22, 0xa3, 0x03, 0x0a, 0x0b, 0x4d, 0x6f, 0x64, 0x75, 0x6c, 0x65, 0x53,
	0x74, 0x61, 0x74, 0x73, 0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x2c, 0x0a, 0x12, 0x70, 0x72, 0x6f, 0x63,
	0x65, 0x73, 0x73, 0x69, 0x6e, 0x67, 0x5f, 0x74, 0x69, 0x6d, 0x65, 0x5f, 0x6d, 0x73, 0x18, 0x02,
	0x20, 0x01, 0x28, 0x04, 0x52, 0x10, 0x70, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x69, 0x6e, 0x67,
	0x54, 0x69, 0x6d, 0x65, 0x4d, 0x73, 0x12, 0x35, 0x0a, 0x17, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x5f,
	0x6f, 0x70, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x74, 0x69, 0x6d, 0x65, 0x5f, 0x6d,
	0x73, 0x18, 0x03, 0x20, 0x01, 0x28, 0x04, 0x52, 0x14, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x4f, 0x70,
	0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x54, 0x69, 0x6d, 0x65, 0x4d, 0x73, 0x12, 0x28, 0x0a,
	0x10, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x5f, 0x72, 0x65, 0x61, 0x64, 0x5f, 0x63, 0x6f, 0x75, 0x6e,
	0x74, 0x18, 0x04, 0x20, 0x01, 0x28, 0x04, 0x52, 0x0e, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x52, 0x65,
	0x61, 0x64, 0x43, 0x6f, 0x75, 0x6e, 0x74, 0x12, 0x61, 0x0a, 0x15, 0x65, 0x78, 0x74, 0x65, 0x72,
	0x6e, 0x61, 0x6c, 0x5f, 0x63, 0x61, 0x6c, 0x6c, 0x5f, 0x6d, 0x65, 0x74, 0x72, 0x69, 0x63, 0x73,
	0x18, 0x05, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x2d, 0x2e, 0x73, 0x66, 0x2e, 0x73, 0x75, 0x62, 0x73,
	0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e, 0x69, 0x6e, 0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0x2e,
	0x76, 0x32, 0x2e, 0x45, 0x78, 0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0x43, 0x61, 0x6c, 0x6c, 0x4d,
	0x65, 0x74, 0x72, 0x69, 0x63, 0x52, 0x13, 0x65, 0x78, 0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0x43,
	0x61, 0x6c, 0x6c, 0x4d, 0x65, 0x74, 0x72, 0x69, 0x63, 0x73, 0x12, 0x2a, 0x0a, 0x11, 0x73, 0x74,
	0x6f, 0x72, 0x65, 0x5f, 0x77, 0x72, 0x69, 0x74, 0x65, 0x5f, 0x63, 0x6f, 0x75, 0x6e, 0x74, 0x18,
	0x0a, 0x20, 0x01, 0x28, 0x04, 0x52, 0x0f, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x57, 0x72, 0x69, 0x74,
	0x65, 0x43, 0x6f, 0x75, 0x6e, 0x74, 0x12, 0x38, 0x0a, 0x18, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x5f,
	0x64, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x70, 0x72, 0x65, 0x66, 0x69, 0x78, 0x5f, 0x63, 0x6f, 0x75,
	0x6e, 0x74, 0x18, 0x0b, 0x20, 0x01, 0x28, 0x04, 0x52, 0x16, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x44,
	0x65, 0x6c, 0x65, 0x74, 0x65, 0x70, 0x72, 0x65, 0x66, 0x69, 0x78, 0x43, 0x6f, 0x75, 0x6e, 0x74,
	0x12, 0x28, 0x0a, 0x10, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x5f, 0x73, 0x69, 0x7a, 0x65, 0x5f, 0x62,
	0x79, 0x74, 0x65, 0x73, 0x18, 0x0c, 0x20, 0x01, 0x28, 0x04, 0x52, 0x0e, 0x73, 0x74, 0x6f, 0x72,
	0x65, 0x53, 0x69, 0x7a, 0x65, 0x42, 0x79, 0x74, 0x65, 0x73, 0x22, 0x57, 0x0a, 0x12, 0x45, 0x78,
	0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0x43, 0x61, 0x6c, 0x6c, 0x4d, 0x65, 0x74, 0x72, 0x69, 0x63,
	0x12, 0x12, 0x0a, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04,
	0x6e, 0x61, 0x6d, 0x65, 0x12, 0x14, 0x0a, 0x05, 0x63, 0x6f, 0x75, 0x6e, 0x74, 0x18, 0x02, 0x20,
	0x01, 0x28, 0x04, 0x52, 0x05, 0x63, 0x6f, 0x75, 0x6e, 0x74, 0x12, 0x17, 0x0a, 0x07, 0x74, 0x69,
	0x6d, 0x65, 0x5f, 0x6d, 0x73, 0x18, 0x03, 0x20, 0x01, 0x28, 0x04, 0x52, 0x06, 0x74, 0x69, 0x6d,
	0x65, 0x4d, 0x73, 0x22, 0x7f, 0x0a, 0x09, 0x43, 0x6f, 0x6d, 0x70, 0x6c, 0x65, 0x74, 0x65, 0x64,
	0x12, 0x57, 0x0a, 0x14, 0x61, 0x6c, 0x6c, 0x5f, 0x70, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x65,
	0x64, 0x5f, 0x72, 0x61, 0x6e, 0x67, 0x65, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x25,
	0x2e, 0x73, 0x66, 0x2e, 0x73, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e, 0x69,
	0x6e, 0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0x2e, 0x76, 0x32, 0x2e, 0x42, 0x6c, 0x6f, 0x63, 0x6b,
	0x52, 0x61, 0x6e, 0x67, 0x65, 0x52, 0x12, 0x61, 0x6c, 0x6c, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73,
	0x73, 0x65, 0x64, 0x52, 0x61, 0x6e, 0x67, 0x65, 0x73, 0x12, 0x19, 0x0a, 0x08, 0x74, 0x72, 0x61,
	0x63, 0x65, 0x5f, 0x69, 0x64, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x74, 0x72, 0x61,
	0x63, 0x65, 0x49, 0x64, 0x22, 0x5b, 0x0a, 0x06, 0x46, 0x61, 0x69, 0x6c, 0x65, 0x64, 0x12, 0x16,
	0x0a, 0x06, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06,
	0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x12, 0x12, 0x0a, 0x04, 0x6c, 0x6f, 0x67, 0x73, 0x18, 0x02,
	0x20, 0x03, 0x28, 0x09, 0x52, 0x04, 0x6c, 0x6f, 0x67, 0x73, 0x12, 0x25, 0x0a, 0x0e, 0x6c, 0x6f,
	0x67, 0x73, 0x5f, 0x74, 0x72, 0x75, 0x6e, 0x63, 0x61, 0x74, 0x65, 0x64, 0x18, 0x03, 0x20, 0x01,
	0x28, 0x08, 0x52, 0x0d, 0x6c, 0x6f, 0x67, 0x73, 0x54, 0x72, 0x75, 0x6e, 0x63, 0x61, 0x74, 0x65,
	0x64, 0x22, 0x4a, 0x0a, 0x0a, 0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x52, 0x61, 0x6e, 0x67, 0x65, 0x12,
	0x1f, 0x0a, 0x0b, 0x73, 0x74, 0x61, 0x72, 0x74, 0x5f, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x18, 0x02,
	0x20, 0x01, 0x28, 0x04, 0x52, 0x0a, 0x73, 0x74, 0x61, 0x72, 0x74, 0x42, 0x6c, 0x6f, 0x63, 0x6b,
	0x12, 0x1b, 0x0a, 0x09, 0x65, 0x6e, 0x64, 0x5f, 0x62, 0x6c, 0x6f, 0x63, 0x6b, 0x18, 0x03, 0x20,
	0x01, 0x28, 0x04, 0x52, 0x08, 0x65, 0x6e, 0x64, 0x42, 0x6c, 0x6f, 0x63, 0x6b, 0x32, 0x7f, 0x0a,
	0x0a, 0x53, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x12, 0x71, 0x0a, 0x0c, 0x50,
	0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x52, 0x61, 0x6e, 0x67, 0x65, 0x12, 0x2e, 0x2e, 0x73, 0x66,
	0x2e, 0x73, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e, 0x69, 0x6e, 0x74, 0x65,
	0x72, 0x6e, 0x61, 0x6c, 0x2e, 0x76, 0x32, 0x2e, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x52,
	0x61, 0x6e, 0x67, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x2f, 0x2e, 0x73, 0x66,
	0x2e, 0x73, 0x75, 0x62, 0x73, 0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2e, 0x69, 0x6e, 0x74, 0x65,
	0x72, 0x6e, 0x61, 0x6c, 0x2e, 0x76, 0x32, 0x2e, 0x50, 0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x52,
	0x61, 0x6e, 0x67, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x30, 0x01, 0x42, 0x4d,
	0x5a, 0x4b, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x73, 0x74, 0x72,
	0x65, 0x61, 0x6d, 0x69, 0x6e, 0x67, 0x66, 0x61, 0x73, 0x74, 0x2f, 0x73, 0x75, 0x62, 0x73, 0x74,
	0x72, 0x65, 0x61, 0x6d, 0x73, 0x2f, 0x70, 0x62, 0x2f, 0x73, 0x66, 0x2f, 0x73, 0x75, 0x62, 0x73,
	0x74, 0x72, 0x65, 0x61, 0x6d, 0x73, 0x2f, 0x69, 0x6e, 0x74, 0x65, 0x72, 0x6e, 0x2f, 0x76, 0x32,
	0x3b, 0x70, 0x62, 0x73, 0x73, 0x69, 0x6e, 0x74, 0x65, 0x72, 0x6e, 0x61, 0x6c, 0x62, 0x06, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_sf_substreams_intern_v2_service_proto_rawDescOnce sync.Once
	file_sf_substreams_intern_v2_service_proto_rawDescData = file_sf_substreams_intern_v2_service_proto_rawDesc
)

func file_sf_substreams_intern_v2_service_proto_rawDescGZIP() []byte {
	file_sf_substreams_intern_v2_service_proto_rawDescOnce.Do(func() {
		file_sf_substreams_intern_v2_service_proto_rawDescData = protoimpl.X.CompressGZIP(file_sf_substreams_intern_v2_service_proto_rawDescData)
	})
	return file_sf_substreams_intern_v2_service_proto_rawDescData
}

var file_sf_substreams_intern_v2_service_proto_msgTypes = make([]protoimpl.MessageInfo, 8)
var file_sf_substreams_intern_v2_service_proto_goTypes = []interface{}{
	(*ProcessRangeRequest)(nil),  // 0: sf.substreams.internal.v2.ProcessRangeRequest
	(*ProcessRangeResponse)(nil), // 1: sf.substreams.internal.v2.ProcessRangeResponse
	(*Update)(nil),               // 2: sf.substreams.internal.v2.Update
	(*ModuleStats)(nil),          // 3: sf.substreams.internal.v2.ModuleStats
	(*ExternalCallMetric)(nil),   // 4: sf.substreams.internal.v2.ExternalCallMetric
	(*Completed)(nil),            // 5: sf.substreams.internal.v2.Completed
	(*Failed)(nil),               // 6: sf.substreams.internal.v2.Failed
	(*BlockRange)(nil),           // 7: sf.substreams.internal.v2.BlockRange
	(*v1.Modules)(nil),           // 8: sf.substreams.v1.Modules
}
var file_sf_substreams_intern_v2_service_proto_depIdxs = []int32{
	8, // 0: sf.substreams.internal.v2.ProcessRangeRequest.modules:type_name -> sf.substreams.v1.Modules
	6, // 1: sf.substreams.internal.v2.ProcessRangeResponse.failed:type_name -> sf.substreams.internal.v2.Failed
	5, // 2: sf.substreams.internal.v2.ProcessRangeResponse.completed:type_name -> sf.substreams.internal.v2.Completed
	2, // 3: sf.substreams.internal.v2.ProcessRangeResponse.update:type_name -> sf.substreams.internal.v2.Update
	3, // 4: sf.substreams.internal.v2.Update.modules_stats:type_name -> sf.substreams.internal.v2.ModuleStats
	4, // 5: sf.substreams.internal.v2.ModuleStats.external_call_metrics:type_name -> sf.substreams.internal.v2.ExternalCallMetric
	7, // 6: sf.substreams.internal.v2.Completed.all_processed_ranges:type_name -> sf.substreams.internal.v2.BlockRange
	0, // 7: sf.substreams.internal.v2.Substreams.ProcessRange:input_type -> sf.substreams.internal.v2.ProcessRangeRequest
	1, // 8: sf.substreams.internal.v2.Substreams.ProcessRange:output_type -> sf.substreams.internal.v2.ProcessRangeResponse
	8, // [8:9] is the sub-list for method output_type
	7, // [7:8] is the sub-list for method input_type
	7, // [7:7] is the sub-list for extension type_name
	7, // [7:7] is the sub-list for extension extendee
	0, // [0:7] is the sub-list for field type_name
}

func init() { file_sf_substreams_intern_v2_service_proto_init() }
func file_sf_substreams_intern_v2_service_proto_init() {
	if File_sf_substreams_intern_v2_service_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_sf_substreams_intern_v2_service_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessRangeRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_sf_substreams_intern_v2_service_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ProcessRangeResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_sf_substreams_intern_v2_service_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Update); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_sf_substreams_intern_v2_service_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ModuleStats); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_sf_substreams_intern_v2_service_proto_msgTypes[4].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ExternalCallMetric); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_sf_substreams_intern_v2_service_proto_msgTypes[5].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Completed); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_sf_substreams_intern_v2_service_proto_msgTypes[6].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Failed); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_sf_substreams_intern_v2_service_proto_msgTypes[7].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*BlockRange); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	file_sf_substreams_intern_v2_service_proto_msgTypes[1].OneofWrappers = []interface{}{
		(*ProcessRangeResponse_Failed)(nil),
		(*ProcessRangeResponse_Completed)(nil),
		(*ProcessRangeResponse_Update)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_sf_substreams_intern_v2_service_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   8,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_sf_substreams_intern_v2_service_proto_goTypes,
		DependencyIndexes: file_sf_substreams_intern_v2_service_proto_depIdxs,
		MessageInfos:      file_sf_substreams_intern_v2_service_proto_msgTypes,
	}.Build()
	File_sf_substreams_intern_v2_service_proto = out.File
	file_sf_substreams_intern_v2_service_proto_rawDesc = nil
	file_sf_substreams_intern_v2_service_proto_goTypes = nil
	file_sf_substreams_intern_v2_service_proto_depIdxs = nil
}
